"""
vpp_qubo.py
========================

This module provides a utility for constructing Quadratic Unconstrained
Binary Optimization (QUBO) formulations of the downlink Vector
Perturbation Precoding (VPP) problem described in the paper *Quantum
Annealing for Large MIMO Downlink Vector Perturbation Precoding*
(`goalpaper.pdf`).  The implementation here mirrors the mathematical
derivations in Section II and Section III A of that paper.  In short,
VPP seeks to find an integer-valued perturbation vector ``v`` that
minimizes the transmit power of a zero‑forcing precoded signal in a
multi‑user MIMO downlink.  That optimization can be expressed as a
quadratic form of complex vectors and subsequently mapped into a
real-valued quadratic form.  Finally, each integer component of the
perturbation is represented by a linear combination of binary
variables, yielding a QUBO.

The high‑level steps are:

  1. Compute the receiver covariance matrix
     ``R = (H @ H.conj().T)**(-1)``, where ``H`` is the ``Nr×Nt``
     complex channel matrix.  For full rank ``H`` (``Nr ≤ Nt``), the
     matrix ``R`` is well defined.

  2. Form the block‑real matrix ``M`` that maps the complex quadratic
     form ``v^H R v`` into the real domain.  Let ``a = Re(v)`` and
     ``b = Im(v)``.  Then

        v^H R v = [a^T b^T] @ M @ [a; b],

     where

        M = [[Re(R), -Im(R)],
             [Im(R),  Re(R)]].

  3. Derive the linear term arising from the cross‑term of the VPP
     objective ``(u+τv)^H R (u+τv)``.  After dropping the constant
     ``u^H R u``, the relevant cost becomes

        τ² v^H R v + 2τ Re(u^H R v).

     In real coordinates this becomes

        w^T (τ² M) w + g^T w,

     where ``w = [Re(v); Im(v)]`` and ``g`` is a 2·Nr‑dimensional real
     vector given by

        g = 2τ · [Re(R).T @ Re(u) + Im(R).T @ Im(u),
                  Re(R).T @ Im(u) − Im(R).T @ Re(u)].

  4. Represent each integer component of ``w`` using ``t+1`` binary
     variables as in Eq. (7) of the VPP paper:

        w_i = ∑_{m=0}^{t-1} 2^m · q_{i,m} − 2^t · q_{i,t},

     where each ``q_{i,m}∈{0,1}``.  This binary encoding allows
     ``w_i`` to take on all integer values in ``[−2^t, 2^t − 1]``.

  5. Construct a linear mapping matrix ``B`` such that ``w = B @ q``
     where ``q`` is the concatenated vector of all binary variables.
     The QUBO cost in terms of ``q`` is

        E(q) = q^T (B^T (τ² M) B + diag(B^T g)) q.

     The diagonal contribution ``diag(B^T g)`` converts the linear
     term ``g^T w`` into linear QUBO biases because ``q_i² = q_i`` for
     binary variables.

The primary function exposed by this module, :func:`build_vpp_qubo`,
computes the QUBO matrix for given channel matrix ``H``, user data
vector ``u``, scaling parameter ``τ``, and bit depth ``t``.  It
returns the symmetric QUBO matrix ``Q`` and a dictionary of
QUBO coefficients suitable for D‑Wave’s API (biases and couplers).

Example usage::

    import numpy as np
    from vpp_qubo import build_vpp_qubo

    # Simple 2×2 MIMO channel and user data
    H = np.array([[1+0.1j, 0.5-0.2j],
                  [0.3+0.4j, 1.2+0.1j]])
    u = np.array([1+1j, -1+0.5j])
    tau = 2.0    # choose based on constellation
    t = 1        # allow perturbation values in [-2, -1, 0, 1]

    Q_matrix, qubo_dict = build_vpp_qubo(H, u, tau=tau, t=t)
    print("QUBO dimension:", Q_matrix.shape)
    print("Number of binary variables:", len(qubo_dict))

References
----------

* S. Kasi, A. K. Singh, D. Venturelli, and K. Jamieson,
  "Quantum Annealing for Large MIMO Downlink Vector Perturbation
  Precoding," 2019, accepted to IEEE ICC 2021.  See Eq. (2) and
  Eq. (7) for the VPP problem formulation and binary encoding.

* M. Kim, D. Venturelli, and K. Jamieson, "Leveraging Quantum
  Annealing for Large MIMO Processing in Centralized Radio Access
  Networks," SIGCOMM 2019.  Appendix A details the derivation of
  similar QUBO forms for maximum likelihood detection in the MIMO
  uplink, which is mathematically dual to the downlink VPP problem
  considered here.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple, List, Union

import numpy as np


def _compute_R_matrix(H: np.ndarray) -> np.ndarray:
    """Compute the inverse of ``H Hᴴ``, i.e. ``(H @ H.conj().T)`` inverse.

    Parameters
    ----------
    H : ndarray of shape (Nr, Nt)
        Complex channel matrix.

    Returns
    -------
    R : ndarray of shape (Nr, Nr)
        Inverse of ``H @ H.conj().T``.

    Notes
    -----
    In zero‑forcing precoding, the transmit precoder is ``P = Hᴴ @ (H Hᴴ)⁻¹``.
    Minimizing the transmit power amounts to minimizing ``dᴴ (H Hᴴ)⁻¹ d``
    where ``d = u + τ v``.
    """
    # Compute Gram matrix at the receiver side
    gram = H @ H.conjugate().T
    # Invert; use pseudo‑inverse if necessary to handle rank‑deficient cases
    R = np.linalg.pinv(gram)
    return R


def _build_block_real_matrix(R: np.ndarray) -> np.ndarray:
    """Construct the real block matrix ``M`` from complex matrix ``R``.

    For a complex Hermitian matrix ``R``, the real mapping matrix

        M = [[Re(R), -Im(R)],
             [Im(R),  Re(R)]],

    satisfies ``v.conj().T @ R @ v = w.T @ M @ w``, where
    ``w = [Re(v); Im(v)]``.

    Parameters
    ----------
    R : ndarray of shape (Nr, Nr)
        Complex Hermitian matrix.

    Returns
    -------
    M : ndarray of shape (2*Nr, 2*Nr)
        Real symmetric matrix for the quadratic term in ``w``.
    """
    Re_R = R.real
    Im_R = R.imag
    # Construct the block matrix
    upper = np.hstack([Re_R, -Im_R])
    lower = np.hstack([Im_R,  Re_R])
    M = np.vstack([upper, lower])
    return M


def _compute_linear_term(R: np.ndarray, u: np.ndarray, tau: float) -> np.ndarray:
    """Compute the linear vector ``g`` for the VPP cost function.

    The linear term arises from the cross term ``2τ Re(uᴴ R v)`` in
    ``(u + τ v)ᴴ R (u + τ v)``.  In real coordinates it can be expressed as

        g = 2τ · [Re(R).T @ Re(u) + Im(R).T @ Im(u),
                  Re(R).T @ Im(u) − Im(R).T @ Re(u)].

    Parameters
    ----------
    R : ndarray of shape (Nr, Nr)
        Complex matrix ``(H Hᴴ)⁻¹``.
    u : ndarray of shape (Nr,)
        Complex user data vector (symbols).
    tau : float
        Scaling constant ``τ = 2(|c_max| + Δ/2)`` controlling the
        integer spacing of perturbation values.

    Returns
    -------
    g : ndarray of shape (2*Nr,)
        Real vector corresponding to linear coefficients in ``w``.
    """
    # Split real and imaginary parts
    c = u.real
    d = u.imag
    Re_R = R.real
    Im_R = R.imag
    # Compute the two halves
    g1 = Re_R.T @ c + Im_R.T @ d
    g2 = Re_R.T @ d - Im_R.T @ c
    # Combine and scale
    g = 2.0 * tau * np.concatenate([g1, g2])
    return g


def _build_binary_mapping(num_vars: int, t: int) -> np.ndarray:
    """Create the linear mapping matrix ``B`` from binary variables to integer values.

    Each real coordinate of the perturbation vector ``w`` is encoded using
    ``t+1`` binary variables ``q_{i,0}, …, q_{i,t}`` via

        w_i = Σ_{m=0}^{t-1} 2^m · q_{i,m} − 2^t · q_{i,t}.

    Here ``num_vars`` is the dimension of ``w`` (``2×Nr``).  The total
    number of binary variables is ``num_vars * (t+1)``.

    Parameters
    ----------
    num_vars : int
        Length of the real perturbation vector ``w`` (equal to 2×Nr).
    t : int
        Bit depth controlling the range of integer values; the encoded
        integers lie in ``[−2^t, 2^t − 1]``.

    Returns
    -------
    B : ndarray of shape (num_vars, num_vars * (t+1))
        Mapping matrix such that ``w = B @ q`` where ``q`` is a vector
        of binary variables.
    """
    total_bits = num_vars * (t + 1)
    B = np.zeros((num_vars, total_bits), dtype=float)
    # For each real coordinate, assign bit weights
    for i in range(num_vars):
        # index offset for this coordinate
        base = i * (t + 1)
        # weights 2^0, …, 2^{t-1} for the positive bits
        for m in range(t):
            B[i, base + m] = 2 ** m
        # weight −2^t for the sign bit
        B[i, base + t] = - (2 ** t)
    return B


def build_vpp_qubo(
    H: np.ndarray,
    u: np.ndarray,
    tau: float,
    t: int = 1,
    *,
    return_matrix: bool = True,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """Construct the QUBO formulation for the downlink VPP problem.

    Given a channel matrix ``H`` and data vector ``u``, this function
    computes the symmetric QUBO matrix ``Q`` whose quadratic form
    ``qᵀ Q q`` encodes the cost of the perturbation vector ``v`` for
    minimizing transmit power in vector perturbation precoding.  The
    binary variables ``q`` are related to the real perturbation values via
    a linear mapping as described in Eq. (7) of the VPP paper.

    Parameters
    ----------
    H : ndarray of shape (Nr, Nt)
        Complex channel matrix.
    u : ndarray of shape (Nr,)
        Complex vector of user data symbols to be precoded.
    tau : float
        Scaling constant ``τ`` that determines the spacing of the
        perturbation lattice.  In VPP this is ``2(|c_max| + Δ/2)``; users
        may choose ``tau`` appropriate for their modulation order.
    t : int, optional (default=1)
        Bit depth of the integer representation.  The perturbation values
        will lie in ``[−2^t, 2^t − 1]``.  Empirically, t=1 suffices for
        common constellations (values in {−2, −1, 0, 1}).
    return_matrix : bool, optional (default=True)
        If ``True``, return both the full symmetric QUBO matrix ``Q`` and
        the dictionary of QUBO coefficients.  If ``False``, return only
        the dictionary of QUBO coefficients.

    Returns
    -------
    Q : ndarray of shape (nq, nq)
        Symmetric QUBO matrix.  Only returned when ``return_matrix`` is
        ``True``.
    qubo : dict mapping (int, int) → float
        Dictionary representation of the QUBO suitable for use with
        D‑Wave’s API.  The keys are pairs ``(i,j)`` with ``i ≤ j``, and
        the values are the corresponding coefficients.  Diagonal entries
        represent linear biases and off‑diagonal entries represent
        couplers.

    Notes
    -----
    The formulation here does not perform any of the pre‑processing
    steps (coefficient scaling and thresholding) described in
    Section III B of the VPP paper.  Those steps can be applied to
    ``Q`` and the returned coefficients as a post‑processing step, if
    desired.
    """
    # Validate dimensions
    Nr, Nt = H.shape
    if u.shape[0] != Nr:
        raise ValueError(f"Dimension mismatch: u has length {u.shape[0]} but H has {Nr} rows")
    # 1. Compute R = (H Hᴴ)⁻¹
    R = _compute_R_matrix(H)
    # 2. Construct block‑real matrix M for vᴴ R v
    M = _build_block_real_matrix(R)
    # 3. Linear term g for cross term
    g = _compute_linear_term(R, u, tau)
    # 4. Binary mapping matrix B
    num_vars = 2 * Nr
    B = _build_binary_mapping(num_vars, t)
    # 5. Compute QUBO matrix: Q = Bᵀ (τ² M) B + diag(Bᵀ g)
    S = (tau ** 2) * M
    Q_quad = B.T @ S @ B
    linear_term = B.T @ g  # yields a vector of length nbits
    Q = Q_quad + np.diag(linear_term)
    # Symmetrize to counter numerical asymmetries
    Q = 0.5 * (Q + Q.T)
    # Build dictionary of coefficients for QUBO
    qubo: Dict[Tuple[int, int], float] = {}
    nbits = Q.shape[0]
    for i in range(nbits):
        for j in range(i, nbits):
            coeff = Q[i, j]
            # Only record non‑zero (or very small) coefficients
            if abs(coeff) > 1e-12:
                qubo[(i, j)] = coeff
    if return_matrix:
        return Q, qubo
    return qubo  # type: ignore[return-value]


def qubo_to_ising(
    qubo: Dict[Tuple[int, int], float]
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Convert a QUBO dictionary into an equivalent Ising form.

    The QUBO energy function is

        E(q) = Σ_i,j Q_{ij} q_i q_j,

    where ``q_i ∈ {0, 1}``.  The corresponding Ising variables are
    defined via the linear change of variables ``s_i = 2 q_i − 1``
    (giving ``s_i ∈ {−1, +1}``).  Substituting and collecting terms
    yields an Ising model

        E(s) = Σ_i h_i s_i + Σ_{i<j} J_{ij} s_i s_j + const.

    This function computes the biases ``h`` and couplers ``J`` from a
    QUBO dictionary.  Constant offsets are omitted because they do
    not affect the optimization.

    Parameters
    ----------
    qubo : dict mapping (int, int) → float
        QUBO coefficients with ``i ≤ j`` keys.

    Returns
    -------
    h : dict mapping int → float
        Biases for Ising variables.
    J : dict mapping (int, int) → float
        Coupling strengths between pairs of Ising variables ``(i, j)`` with
        ``i < j``.
    """
    # Determine number of variables from keys
    if not qubo:
        return {}, {}
    max_idx = max(max(i, j) for i, j in qubo.keys())
    num_vars = max_idx + 1
    # Initialize biases and couplings
    h: Dict[int, float] = {i: 0.0 for i in range(num_vars)}
    J: Dict[Tuple[int, int], float] = {}
    # Convert each QUBO term
    for (i, j), coeff in qubo.items():
        if i == j:
            # Linear term: Q_ii q_i → in Ising: (Q_ii/4) s_i + const
            h[i] += coeff / 4.0
        else:
            # Quadratic term: Q_ij q_i q_j → Ising mapping yields both
            # linear and quadratic contributions:
            # q_i q_j = (1/4)(s_i s_j + s_i + s_j + 1)
            h[i] += coeff / 4.0
            h[j] += coeff / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + coeff / 4.0
    return h, J


if __name__ == "__main__":  # pragma: no cover
    # Example usage when run as a script
    import argparse

    parser = argparse.ArgumentParser(description="Generate QUBO for downlink VPP.")
    parser.add_argument("Hfile", help="Path to a NumPy .npz file containing H (channel matrix)")
    parser.add_argument("ufile", help="Path to a NumPy .npz file containing u (data vector)")
    parser.add_argument("tau", type=float, help="Tau scaling parameter")
    parser.add_argument("t", type=int, nargs="?", default=1, help="Bit depth t (default=1)")
    args = parser.parse_args()

    # Load input arrays; expecting variables 'H' and 'u' inside the npz files
    with np.load(args.Hfile) as fH:
        H = fH['H']
    with np.load(args.ufile) as fu:
        u = fu['u']

    Q, qubo = build_vpp_qubo(H, u, args.tau, t=args.t, return_matrix=True)
    print("Generated QUBO with", Q.shape[0], "binary variables.")
    # Print a few coefficients for inspection
    for idx, (key, val) in enumerate(qubo.items()):
        print(f"QUBO coeff {key}: {val:.4f}")
        if idx > 20:
            break