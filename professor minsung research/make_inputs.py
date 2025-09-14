import numpy as np
H = np.array([[1+0.1j, 0.5-0.2j],
              [0.3+0.4j, 1.2+0.1j]])
u = np.array([1+1j, -1+0.5j])
np.savez("Hfile.npz", H=H)
np.savez("ufile.npz", u=u)
from vpp_qubo import build_vpp_qubo

H = np.array([[1+0.1j, 0.5-0.2j],
              [0.3+0.4j, 1.2+0.1j]])
u = np.array([1+1j, -1+0.5j])

Q, qubo = build_vpp_qubo(H, u, tau=2.0, t=1, return_matrix=True)
print("Q shape:", Q.shape)
