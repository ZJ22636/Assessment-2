import numpy as np
import pytest
import LebwohlLasher as LL
import Cython_LebwohlLasher as CY

N = 20

#Check same results as original
def test_all_energy_matches_original():
    np.random.seed(4)
    lattice = LL.initdat(N)
    assert CY.all_energy(lattice, N) == pytest.approx(LL.all_energy(lattice, N))
    
def test_get_order_matches_original():
    np.random.seed(5)
    lattice = LL.initdat(N)
    assert CY.get_order(lattice, N) == pytest.approx(LL.get_order(lattice, N).real)
    
def test_aligned_lattice():
    lattice = np.zeros((N, N))
    assert CY.all_energy(lattice, N) == pytest.approx(-4.0 * N *N)
    assert CY.get_order(lattice, N) == pytest.approx(1.0)
    
def test_edge_cells_wrap_round():
    np.random.seed(6)
    lattice = LL.initdat(N)
    #corners
    for ix, iy in [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)]:
        assert CY.one_energy(lattice, ix, iy, N) == pytest.approx(LL.one_energy(lattice, ix, iy, N))
        
#MCS tests
def test_acceptence_ratio_range():
    np.random.seed(0)
    ratio = CY.MC_step(CY.initdat(N), 0.5, N)
    assert 0.0 <= ratio <= 1.0
    
def test_mc_step_reproduce():
    np.random.seed(42)
    a = CY.initdat(N)
    CY.MC_step(a, 0.5, N)
    np.random.seed(42)
    b = CY.initdat(N)
    CY.MC_step(b, 0.5, N)
    assert np.array_equal(a, b)
    
#Physics Check
def equilibrium_average(module, temp, seed):
    np.random.seed(seed)
    lattice = module.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = module.MC_step(lattice, temp, N)
        if step >= 150:
            energies.append(CY.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    return np.mean(energies), np.mean(ratios)
    
@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original(temp):
    e_orig, r_orig = equilibrium_average(LL, temp, seed=1)
    e_cy, r_cy = equilibrium_average(CY, temp, seed=1)
    assert e_cy == pytest.approx(e_orig, abs=0.15)   #energy per cell matches  
    assert r_cy == pytest.approx(r_orig, abs=0.03) #acceptence ratio matches
    
#whole program
def test_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    CY.main("test", 5, N, 0.5, 0)
    files = list((tmp_path / "Outputs" / "Cython").glob("*.txt"))
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6,4)
    assert "Timings" in capsys.readouterr().out
    

