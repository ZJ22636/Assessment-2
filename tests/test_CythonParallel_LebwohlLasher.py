import os
import subprocess
import sys
import pathlib
import numpy as np
import pytest
import LebwohlLasher as LL  
import CythonParallel_LebwohlLasher as CYP

N = 20

#same answers as og
def test_all_energy_matches_original():
    np.random.seed(4)
    lattice = LL.initdat(N)
    assert CYP.all_energy(lattice, N) == pytest.approx(LL.all_energy(lattice, N))


def test_get_order_matches_original():
    np.random.seed(5)
    lattice = LL.initdat(N)
    assert CYP.get_order(lattice, N) == pytest.approx(LL.get_order(lattice, N).real)


def test_aligned_lattice():
    lattice = np.zeros((N, N))
    assert CYP.all_energy(lattice, N) == pytest.approx(-4.0 * N * N)
    assert CYP.get_order(lattice, N) == pytest.approx(1.0)
    
#Checkerboard MCS
def test_odd_size_rejected():
    with pytest.raises(ValueError):
        CYP.MC_step(CYP.initdat(5), 0.5, 5)


def test_acceptance_ratio_range():
    np.random.seed(0)
    ratio = CYP.MC_step(CYP.initdat(N), 0.5, N)
    assert 0.0 <= ratio <= 1.0


def test_same_result_any_thread_count():
    threads = os.cpu_count()                         # all available threads
    results = []
    for t in (1, threads):
        CYP.set_threads(t)
        np.random.seed(7)
        lattice = CYP.initdat(N)
        for _ in range(10):
            CYP.MC_step(lattice, 0.5, N)
        results.append(lattice)
    assert np.allclose(results[0], results[1]) 
    
def test_low_temperature_orders():
    np.random.seed(0)
    lattice = CYP.initdat(10)
    start_energy = CYP.all_energy(lattice, 10)
    for _ in range(100):
        CYP.MC_step(lattice, 0.1, 10)
    assert CYP.all_energy(lattice, 10) < start_energy
    assert CYP.get_order(lattice, 10) > 0.8


def test_high_temperature_stays_disordered():
    np.random.seed(0)
    lattice = CYP.initdat(N)
    for _ in range(100):
        CYP.MC_step(lattice, 2.0, N)
    assert CYP.get_order(lattice, N) < 0.5
    
#Same physics check
def equilibrium_averages(module, temp, seed):
    np.random.seed(seed)
    lattice = module.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = module.MC_step(lattice, temp, N)
        if step >= 150:                                 
            energies.append(CYP.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    return np.mean(energies), np.mean(ratios)

@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original(temp):
    e_orig, r_orig = equilibrium_averages(LL, temp, seed=1)
    e_cyp, r_cyp = equilibrium_averages(CYP, temp, seed=1)
    assert e_cyp == pytest.approx(e_orig, abs=0.15)
    assert r_cyp == pytest.approx(r_orig, abs=0.03)
    
#Whole program
def test_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    CYP.main("test", 5, N, 0.5, 0, 1)
    files = list((tmp_path / "Outputs" / "CythonParallel").glob("*.txt"))
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6,4)
    assert "Threads: 1" in capsys.readouterr().out