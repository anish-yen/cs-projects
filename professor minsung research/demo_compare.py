import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping

# --- Step 1: Load your saved H and u ---
with np.load("Hfile.npz") as fH:
    H = fH["H"]
with np.load("ufile.npz") as fU:
    u = fU["u"]

tau = 2.0
t = 1
Nr, Nt = H.shape

# --- Step 2: Build QUBO ---
Q, qubo = build_vpp_qubo(H, u, tau, t=t, return_matrix=True)
B = _build_binary_mapping(2 * Nr, t)

print("QUBO matrix shape:", Q.shape)
print("Number of binary variables:", Q.shape[0])

# --- Step 3: Define helpers for energy comparison ---
def vpp_energy_eq2(H, u, tau, v):
    """Direct evaluation of Eq. (2)."""
    R = np.linalg.pinv(H @ H.conjugate().T)
    z = u + tau * v
    x = H.conjugate().T @ (R @ z)
    return float(np.vdot(x, x).real)

def qubo_energy(Q, q):
    return float(q @ Q @ q)

def w_to_v(w):
    """Map real vector w = [Re(v); Im(v)] to complex v."""
    Nr = w.shape[0] // 2
    return w[:Nr] + 1j * w[Nr:]

# constant offset (uᴴRu), drop it for fair comparison
R = np.linalg.pinv(H @ H.conjugate().T)
const = float((u.conjugate().T @ (R @ u)).real)

# --- Step 4: Sample some random q vectors ---
rng = np.random.default_rng(123)
for trial in range(5):
    q = rng.integers(0, 2, size=Q.shape[0]).astype(float)
    w = B @ q
    v = w_to_v(w)

    E_vpp = vpp_energy_eq2(H, u, tau, v) - const
    E_qubo = qubo_energy(Q, q)

    print(f"Trial {trial}: E_vpp={E_vpp:.6f}, E_qubo={E_qubo:.6f}, diff={E_vpp - E_qubo:.2e}")
