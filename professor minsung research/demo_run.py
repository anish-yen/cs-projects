import numpy as np
from vpp_qubo import build_vpp_qubo

# Example: 2x2 channel, 2 users
H = np.array([[1+0.1j, 0.5-0.2j],
              [0.3+0.4j, 1.2+0.1j]])
u = np.array([1+1j, -1+0.5j])
tau = 2.0
t = 1

Q, qubo = build_vpp_qubo(H, u, tau, t=t, return_matrix=True)

print("QUBO matrix shape:", Q.shape)
print("First 5 QUBO entries:")
for i, (key, val) in enumerate(qubo.items()):
    print(f"{key}: {val:.4f}")
    if i >= 4:
        break
