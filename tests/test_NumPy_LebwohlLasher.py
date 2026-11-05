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

#Checker Board MCS
def test_odd_size_rejected():
    with pytest.raises(ValueError):
        NP.MC_step(NP.initdat(5), 0.5, 5)
        
def test_acceptance_ratio_range():
    np.random.seed(0)
    ratio = NP.MC_step(NP.initdat(N), 0.5, N)
    assert 0.0 <= ratio <= 1.0
    
def test_mc_step_reproducible():
    np.random.seed(42)
    a = NP.initdat(N)
    NP.MC_step(a, 0.5, N)
    np.random.seed(42)
    b = NP.initdat(N)
    NP.MC_step(b, 0.5, N)
    assert np.array_equal(a, b)


def test_low_temperature_orders():
    np.random.seed(0)
    lattice = NP.initdat(10)
    start_energy = NP.all_energy(lattice, 10)
    for _ in range(100):
        NP.MC_step(lattice, 0.1, 10)
    assert NP.all_energy(lattice, 10) < start_energy
    assert NP.get_order(lattice, 10) > 0.8


def test_high_temperature_stays_disordered():
    np.random.seed(0)
    lattice = NP.initdat(N)
    for _ in range(100):
        NP.MC_step(lattice, 2.0, N)
    assert NP.get_order(lattice, N) < 0.5
    
#Check physics good
def equilibrium_averages(module, temp, seed):
    np.random.seed(seed)
    lattice = module.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = module.MC_step(lattice, temp, N)
        if step >= 150: # skip the equilibration part
            energies.append(NP.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    return np.mean(energies), np.mean(ratios)


@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original(temp):
    e_orig, r_orig = equilibrium_averages(LL, temp, seed=1)
    e_np, r_np = equilibrium_averages(NP, temp, seed=1)
    assert e_np == pytest.approx(e_orig, abs=0.15) # energy per cell agrees
    assert r_np == pytest.approx(r_orig, abs=0.03) # acceptance ratio agrees

#Whole Program
def test_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    NP.main("test", 5, N, 0.5, 0)
    files = list((tmp_path /"Outputs" / "NumPy").glob("*.txt")) #check saved right place
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6,4)
    assert "Timings" in capsys.readouterr().out