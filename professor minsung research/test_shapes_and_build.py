# test_shapes_and_build.py
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping
from test_helpers import random_channel, random_symbols

def test_qubo_shapes_multiple_sizes():
    rng = np.random.default_rng(0)
    for Nr, Nt, t in [(1,2,1), (2,3,1), (3,4,2), (5,6,1)]:
        H = random_channel(Nr, Nt, seed=int(rng.integers(1, 1e6)))
        u = random_symbols(Nr, seed=int(rng.integers(1, 1e6)))
        tau = 2.0
        Q, qubo = build_vpp_qubo(H, u, tau, t=t)

        n_bits = (t+1) * 2 * Nr
        assert Q.shape == (n_bits, n_bits)
        # diagonal must be finite; off-diagonals symmetric numerically
        assert np.all(np.isfinite(Q))
        assert np.allclose(Q, (Q + Q.T)/2, atol=1e-10)

        # mapping matrix shape sanity
        B = _build_binary_mapping(2*Nr, t=t)
        assert B.shape == (2*Nr, n_bits)
