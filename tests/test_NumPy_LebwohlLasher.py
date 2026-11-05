import numpy as np
import pytest
import LebwohlLasher as LL
import NumPy_LebwohlLasher as NP


N = 20


def test_cell_energies_match_one_energy():
    np.random.seed(3)
    lattice = LL.initdat(N)
    original = np.array([[LL.one_energy(lattice, i, j, N) for j in range(N)] for i in range(N)])
    assert NP.cell_energies(lattice) == pytest.approx(original)


def test_all_energy_matches_original():
    np.random.seed(4)
    lattice = LL.initdat(N)
    assert NP.all_energy(lattice, N) == pytest.approx(LL.all_energy(lattice, N))
    
def test_get_order_matches_original():
    np.random.seed(5)
    lattice = LL.initdat(N)
    assert NP.get_order(lattice, N) == pytest.approx(LL.get_order(lattice, N).real)


def test_get_order_is_real():
    np.random.seed(6)
    assert np.isrealobj(NP.get_order(LL.initdat(N), N))   # eigh never gives complex numbers