# test_smoke.py
import numpy as np
from vpp_qubo import build_vpp_qubo

np.random.seed(0)

# tiny system so we can enumerate later if we want
Nr, Nt = 2, 3
H = np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)
u = (np.random.randn(Nr) + 1j*np.random.randn(Nr))  # any complex symbols
tau = 2  # e.g., QAM spacing-based constant, OK for a test

# build QUBO with t=1 (allows -2..+1, covering {-1,0,1})
Q, qubo = build_vpp_qubo(H, u, tau, t=1)

print("Q shape:", Q.shape)
print("Non-zeros in QUBO dict:", len(qubo))
