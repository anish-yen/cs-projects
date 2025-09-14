# test_random_spotchecks.py
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping
from test_helpers import vpp_energy_eq2, qubo_energy, w_from_q, w_to_v, random_channel, random_symbols, vpp_constant_offset

def test_random_spotchecks_match_eq2():
    rng = np.random.default_rng(123)
    cases = [(2,3,1), (3,5,1), (4,6,1), (2,3,2)]

    for Nr, Nt, t in cases:
        H = random_channel(Nr, Nt, seed=int(rng.integers(1, 1e6)))
        u = random_symbols(Nr, seed=int(rng.integers(1, 1e6)))
        tau = 2.0

        Q, _ = build_vpp_qubo(H, u, tau, t=t)
        B = _build_binary_mapping(2*Nr, t=t)
        n_bits = B.shape[1]

        const = vpp_constant_offset(H, u)

        for _ in range(200):
            q = rng.integers(0, 2, size=n_bits).astype(float)
            w = w_from_q(B, q)
            v = w_to_v(w)

            E_vpp  = vpp_energy_eq2(H, u, tau, v) - const
            E_qubo = qubo_energy(Q, q)
            assert abs(E_vpp - E_qubo) < 1e-9
