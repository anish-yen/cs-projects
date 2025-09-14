# test_correctness.py
import itertools
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping

def complex_to_real_block(R):
    # Real block form [[Re, -Im],[Im, Re]]
    return np.block([[R.real, -R.imag],
                     [R.imag,  R.real]])

def vpp_cost(H, u, tau, v):
    """Original VPP objective: || H^H (H H^H)^(-1) (u + tau v) ||^2"""
    R = np.linalg.pinv(H @ H.conjugate().T)
    z = u + tau * v
    x = H.conjugate().T @ (R @ z)
    return float(np.vdot(x, x).real)

def qubo_energy(Q, q):
    # q is 0/1 vector; energy is q^T Q q
    return float(q @ Q @ q)

np.random.seed(1)

# tiny instance: Nr=1, Nt=2 so we can enumerate fast
Nr, Nt = 1, 2
H = np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)
u = np.random.randn(Nr) + 1j*np.random.randn(Nr)
tau = 2
t = 1  # {-2,-1,0,1} range

# build QUBO
Q, qubo = build_vpp_qubo(H, u, tau, t=t)

# mapping from q -> integer perturbation w = [a1,b1] since Nr=1
B = _build_binary_mapping(2*Nr, t=t)  # shape (2*Nr, n_bits)
n_bits = B.shape[1]

# enumerate all 2^(n_bits) bitstrings
max_err = 0.0
best_vpp = None
best_q = None

for bits in itertools.product([0,1], repeat=n_bits):
    q = np.array(bits, dtype=float)

    # q -> real perturb vector w (length 2*Nr), then to complex v (length Nr)
    w = B @ q
    v = w[0::2] + 1j*w[1::2]

    # energies
    E_vpp  = vpp_cost(H, u, tau, v)
    E_qubo = qubo_energy(Q, q)

    max_err = max(max_err, abs(E_vpp - E_qubo))
    if best_vpp is None or E_vpp < best_vpp:
        best_vpp = E_vpp
        best_q = q

print(f"max |VPP cost - QUBO energy| over all configs = {max_err:.6e}")
print(f"best VPP energy found       = {best_vpp:.6f}")
print(f"argmin bitstring (first 16) = {best_q[:16].astype(int)}")
