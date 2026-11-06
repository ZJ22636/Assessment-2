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
    
#whole program
def test_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    CY.main("test", 5, N, 0.5, 0)
    files = list((tmp_path / "Outputs" / "Cython").glob("*.txt"))
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6,4)
    assert "Timings" in capsys.readouterr().out
    

