# test_helpers.py
import numpy as np

def complex_to_real_block(R: np.ndarray) -> np.ndarray:
    """Return real block matrix [[Re R, -Im R],[Im R, Re R]]."""
    return np.block([[R.real, -R.imag],[R.imag, R.real]])

def vpp_energy_eq2(H: np.ndarray, u: np.ndarray, tau: float, v: np.ndarray) -> float:
    """
    Source-of-truth energy from Eq (2):
        E(v) = || H^H (H H^H)^(-1) (u + tau v) ||^2
    """
    R = np.linalg.pinv(H @ H.conjugate().T)
    z = u + tau * v
    x = H.conjugate().T @ (R @ z)
    return float(np.vdot(x, x).real)

def qubo_energy(Q: np.ndarray, q_bits: np.ndarray) -> float:
    """QUBO energy q^T Q q for binary vector q_bits."""
    q = q_bits.astype(float)
    return float(q @ Q @ q)

def w_from_q(B: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """Map binary q -> real perturbation vector w=[Re(v1),Im(v1),...,Re(vNr),Im(vNr)]."""
    return B @ q_bits.astype(float)

def w_to_v(w: np.ndarray) -> np.ndarray:
    """Pack w = [Re(v1..vNr), Im(v1..vNr)] into complex v."""
    Nr = w.shape[0] // 2
    return w[:Nr] + 1j * w[Nr:]

def random_channel(Nr: int, Nt: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((Nr, Nt)) + 1j * rng.standard_normal((Nr, Nt))

def random_symbols(Nr: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr)
# 
def vpp_constant_offset(H: np.ndarray, u: np.ndarray) -> float:
    R = np.linalg.pinv(H @ H.conjugate().T)
    return float((u.conjugate().T @ (R @ u)).real)
