# diag_g_check.py
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_block_real_matrix, _compute_R_matrix

np.random.seed(1)
Nr, Nt = 1, 2
H = np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)
u = np.random.randn(Nr) + 1j*np.random.randn(Nr)
tau = 2.0

R = _compute_R_matrix(H)
M = _build_block_real_matrix(R)
u_prime = np.concatenate([u.real, u.imag])

g_expected = 2.0 * tau * (M @ u_prime)

# Pull g as used inside build_vpp_qubo by rebuilding pieces
# (temporarily mirror lines in build_vpp_qubo here if needed)
print("g_expected:", g_expected)
