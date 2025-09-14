"""
vpp_qubo.py
========================

This constructs Quadratic Unconstrained Binary Optimization (QUBO)
for downlink Vector Perturbation Precoding (VPP)

  • Original VPP objective (Eq. 2):
      v* = argmin_v || Hᴴ (H Hᴴ)⁻¹ (u + τ v) ||²
    This is the NP-hard transmit-power minimization over an integer (Gaussian)
    perturbation v that improves zero-forcing precoding.

  • Real quadratic form (our Eq. 4 template): write R = (H Hᴴ)⁻¹ and convert
    complex quadratic to a real block form with w = [Re(v); Im(v)],
      E(w) = wᵀ (τ² M) w + gᵀ w + const,
    where M = [[Re(R), -Im(R)],[Im(R), Re(R)]] and g encodes 2τ·Re(uᴴ R v).
    (This is the standard complex→real embedding used before we binary-encode
    the integers).

  • Binary integer encoding (Eq. 7): each real/imag entry of v is mapped using
      w_i = Σ_{m=0}^{t-1} 2^m q_{i,m} - 2^t q_{i,t},   q_{i,m} ∈ {0,1},
    so w_i ∈ [−2^t, 2^t − 1]. With t=1 we already cover {−1,0,1} (and ±2) which
    is typically sufficient for VPP.

  • Final QUBO (Eq. 8): substitute w = B q into the real quadratic to obtain
      E(q) = qᵀ [ Bᵀ (τ² M) B ] q + (Bᵀ g)ᵀ q + const
            = qᵀ Q q  with  Q = Bᵀ (τ² M) B + diag(Bᵀ g),
    where diagonal entries act as linear biases f_i and off-diagonals as couplers
    g_{ij} in the paper’s notation.


"""


from __future__ import annotations

import itertools
from typing import Dict, Tuple, List, Union

import numpy as np


def _compute_R_matrix(H: np.ndarray) -> np.ndarray:
    # Step 1 (Eq. 2): build R = (H Hᴴ)⁻¹
    # --------------------------------------------------
    # In the paper, Eq. (2) says the transmit power depends on
    # (u + τv)ᴴ (H Hᴴ)⁻¹ (u + τv).
    # So the very first thing we need is that matrix R = (H Hᴴ)⁻¹.
    # If H is not full rank, we fall back to a pseudo-inverse so that
    # the math still works numerically.

    # Compute Gram matrix at the receiver side
    gram = H @ H.conjugate().T
    # Invert; use pseudo‑inverse if necessary to handle rank‑deficient cases
    R = np.linalg.pinv(gram)
    return R


def _build_block_real_matrix(R: np.ndarray) -> np.ndarray:
    # Step 2: turn complex math into real math
    # --------------------------------------------------
    # Eq. (2) is written with complex vectors. To make it usable in QUBO,
    # we split v into its real and imaginary parts: v = a + j b.
    # Then we stack them into w = [a; b].
    # With this trick, vᴴ R v turns into wᵀ M w,
    # where M is a simple block matrix made of Re(R) and Im(R).
    # This is the standard way to convert complex quadratics into real ones.

    Re_R = R.real
    Im_R = R.imag
    # Construct the block matrix
    upper = np.hstack([Re_R, -Im_R])
    lower = np.hstack([Im_R,  Re_R])
    M = np.vstack([upper, lower])
    return M


def _compute_linear_term(R: np.ndarray, u: np.ndarray, tau: float) -> np.ndarray:
    # Step 3: grab the "cross term" 2τ·Re(uᴴ R v)
    # --------------------------------------------------
    # When you expand (u + τv)ᴴ R (u + τv), three pieces pop out:
    #   • uᴴ R u   (a constant we can ignore),
    #   • 2τ Re(uᴴ R v)  <-- this is the linear term in v,
    #   • τ² vᴴ R v      (the quadratic term we already handled with M).
    # This function computes that middle piece, but written in terms of
    # real vectors, so we can later plug it into the QUBO.

    # Split real and imaginary parts
    c = u.real
    d = u.imag

    Re_R = R.real
    Im_R = R.imag

    # Compute M @ u' without explicitly forming M:
    # top block: Re(R) @ c  - Im(R) @ d
    # bot block: Im(R) @ c  + Re(R) @ d
    top = Re_R @ c - Im_R @ d
    bot = Im_R @ c + Re_R @ d
    u_real = np.concatenate([top, bot])  # this equals (M @ u')

    g = 2.0 * tau * u_real
    return g


def _build_binary_mapping(num_vars: int, t: int) -> np.ndarray:
    # Step 4 (Eq. 7): represent each integer with bits
    # --------------------------------------------------
    # Each entry of w must be an integer. Eq. (7) in the paper shows
    # how to encode it using binary variables:
    #   w_i = sum_{m=0}^{t-1} 2^m q_{i,m} - 2^t q_{i,t}
    # So with t=1, you get values in { -2, -1, 0, 1 }.
    # This function just builds a big matrix B so that w = B q,
    # where q is the vector of all binary bits. That way, the mapping
    # is linear and easy to apply.

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
    # Step 5 (Eq. 8): put it all together → final QUBO
    # --------------------------------------------------
    # By now we have:
    #   • Quadratic part: τ² wᵀ M w
    #   • Linear part: gᵀ w
    #   • Mapping: w = B q
    #
    # Substitute w = B q:
    #   E(q) = qᵀ [Bᵀ (τ² M) B] q  +  (Bᵀ g)ᵀ q
    #
    # To make this pure QUBO, we move the linear term onto the diagonal
    # (since q_i² = q_i for binary). That gives:
    #   Q = Bᵀ (τ² M) B + diag(Bᵀ g)
    #
    # • The diagonal entries of Q are the "biases" f_i.
    # • The off-diagonals are the "couplers" g_{ij}.
    # This is exactly Eq. (8) in the paper.

    # Validate dimensions
    Nr, Nt = H.shape
    if u.shape[0] != Nr:
        raise ValueError(f"Dimension mismatch: u has length {u.shape[0]} but H has {Nr} rows")
    # 1. Compute R = (H Hᴴ)⁻¹
    R = _compute_R_matrix(H)
    # 2. Construct block‑real matrix M for vᴴ R v
    M = _build_block_real_matrix(R)
    
    # 3. Linear term g for cross term (compute directly from M to avoid drift)
    #    E(w) = τ² wᵀ M w + (2τ M u')ᵀ w + const
    u_prime = np.concatenate([u.real, u.imag])        # shape (2*Nr,)
    g = 2.0 * tau * (M @ u_prime)                     # shape (2*Nr,)

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
    # Optional: convert QUBO → Ising form
    # --------------------------------------------------
    # Some solvers prefer spins s_i ∈ {−1, +1} instead of bits q_i ∈ {0,1}.
    # The change of variables is simple: s_i = 2 q_i − 1.
    # After substitution, you get the familiar Ising form:
    #   E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + const
    # where h_i are the linear biases and J_ij are the couplings.
    # This is standard, and is the same trick used in the uplink MLD paper.

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