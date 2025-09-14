# test_separable_channel.py (replacement)
import numpy as np
from vpp_qubo import build_vpp_qubo, _build_binary_mapping

def test_diagonal_channel_is_block_separable_across_coordinates():
    Nr, Nt, t = 2, 4, 1
    rng = np.random.default_rng(42)

    # Make H with orthonormal rows => HH^H = I
    A = rng.standard_normal((Nt, Nt)) + 1j * rng.standard_normal((Nt, Nt))
    Qfull, _ = np.linalg.qr(A)        # Nt x Nt unitary
    H = Qfull[:Nr, :]                 # Nr x Nt, rows orthonormal

    u = rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr)
    tau = 2.0

    Qmat, _ = build_vpp_qubo(H, u, tau, t=t)
    B = _build_binary_mapping(2*Nr, t=t)

    # Build index groups for each real coordinate (there are 2*Nr of them)
    n_bits_per_coord = t + 1
    groups = []
    for coord in range(2*Nr):
        base = coord * n_bits_per_coord
        groups.append(list(range(base, base + n_bits_per_coord)))

    # For any two different coordinates, all couplings between their bit-blocks should be ~0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            block = Qmat[np.ix_(groups[i], groups[j])]
            assert np.allclose(block, 0.0, atol=1e-10)
