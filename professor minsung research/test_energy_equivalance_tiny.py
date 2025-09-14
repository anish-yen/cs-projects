# test_energy_equivalence_tiny.py
import itertools
import numpy as np
from test_helpers import vpp_constant_offset

from vpp_qubo import build_vpp_qubo, _build_binary_mapping
from test_helpers import vpp_energy_eq2, qubo_energy, w_from_q, w_to_v, random_channel, random_symbols

def test_full_enumeration_matches_eq2():
    Nr, Nt, t = 1, 2, 1
    H = random_channel(Nr, Nt, seed=1)
    u = random_symbols(Nr, seed=2)
    tau = 2.0

    Q, _ = build_vpp_qubo(H, u, tau, t=t)
    B = _build_binary_mapping(2*Nr, t=t)
    n_bits = B.shape[1]

    # subtract this constant from Eq.2 energies
    const = vpp_constant_offset(H, u)

    max_abs_err = 0.0
    for bits in itertools.product([0,1], repeat=n_bits):
        q = np.array(bits, dtype=float)
        w = w_from_q(B, q)
        v = w_to_v(w)
        E_vpp  = vpp_energy_eq2(H, u, tau, v) - const
        E_qubo = qubo_energy(Q, q)
        ...


    

    assert max_abs_err < 1e-10, f"Max |Eq2 - QUBO| = {max_abs_err}"
