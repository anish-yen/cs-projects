# test_bitdepth_monotonicity.py
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping
from test_helpers import vpp_energy_eq2, qubo_energy, w_from_q, w_to_v, random_channel, random_symbols

def brute_force_min_energy(H, u, tau, t):
    B = _build_binary_mapping(2*H.shape[0], t=t)
    n_bits = B.shape[1]
    Q, _ = build_vpp_qubo(H, u, tau, t=t)

    # For t up to 2 on small Nr, enumerating 2^n is OK; we only use it for tiny Nr
    best = np.inf
    argmin_q = None
    for i in range(1 << n_bits):
        q = np.array([(i >> k) & 1 for k in range(n_bits)], dtype=float)
        E = qubo_energy(Q, q)
        if E < best:
            best, argmin_q = E, q
    w = w_from_q(B, argmin_q)
    v = w_to_v(w)
    return best, v, argmin_q

def test_increasing_t_never_worsens_best_energy():
    # Tiny case so we can brute-force both t=1 and t=2
    Nr, Nt = 1, 2
    H = random_channel(Nr, Nt, seed=777)
    u = random_symbols(Nr, seed=888)
    tau = 2.0

    E1, v1, q1 = brute_force_min_energy(H, u, tau, t=1)
    E2, v2, q2 = brute_force_min_energy(H, u, tau, t=2)
    # t=2 search space strictly contains t=1 space -> best energy cannot increase
    assert E2 <= E1 + 1e-12
