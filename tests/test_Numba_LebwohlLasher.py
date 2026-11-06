import numpy as np
import pytest
import numba
import LebwohlLasher as LL
import Numba_LebwohlLasher as NB
import NumbaParallel_LebwohlLasher as NBP

N = 20
MODULES = [NB, NBP]

def seed_all(module,seed):
    np.random.seed(seed)
    module.set_seed(seed)
    
#Check matches original
@pytest.mark.parametrize("module", MODULES)
def test_all_energy_matches_original(module):
    np.random.seed(4)
    lattice = LL.initdat(N)
    assert module.all_energy(lattice, N) == pytest.approx(LL.all_energy(lattice, N))


@pytest.mark.parametrize("module", MODULES)
def test_get_order_matches_original(module):
    np.random.seed(5)
    lattice = LL.initdat(N)
    assert module.get_order(lattice, N) == pytest.approx(LL.get_order(lattice, N).real)


@pytest.mark.parametrize("module", MODULES)
def test_aligned_lattice(module):
    lattice = np.zeros((N, N))
    assert module.all_energy(lattice, N) == pytest.approx(-4.0 * N * N)
    assert module.get_order(lattice, N) == pytest.approx(1.0)
    
#MCS behaviour
@pytest.mark.parametrize("module", MODULES)
def test_acceptance_ratio_range(module):
    seed_all(module, 0)
    ratio = module.MC_step(module.initdat(N), 0.5, N)
    assert 0.0 <= ratio <= 1.0


@pytest.mark.parametrize("module", MODULES)
def test_mc_step_reproducible(module):
    seed_all(module, 42)
    a = module.initdat(N)
    module.MC_step(a, 0.5, N)
    seed_all(module, 42)
    b = module.initdat(N)
    module.MC_step(b, 0.5, N)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("module", MODULES)
def test_low_temperature_orders(module):
    seed_all(module, 0)
    lattice = module.initdat(10)
    start_energy = module.all_energy(lattice, 10)
    for _ in range(100):
        module.MC_step(lattice, 0.1, 10)
    assert module.all_energy(lattice, 10) < start_energy
    assert module.get_order(lattice, 10) > 0.8


@pytest.mark.parametrize("module", MODULES)
def test_high_temperature_stays_disordered(module):
    seed_all(module, 0)
    lattice = module.initdat(N)
    for _ in range(100):
        module.MC_step(lattice, 2.0, N)
    assert module.get_order(lattice, N) < 0.5
    
#Parallel only
def test_parallel_odd_size_rejected():
    with pytest.raises(ValueError):
        NBP.MC_step(NBP.initdat(5), 0.5, 5) 
        
def test_parallel_same_result_any_thread_count():
    threads = numba.config.NUMBA_NUM_THREADS          # all available threads
    results = []
    for t in (1, threads):
        numba.set_num_threads(t)
        seed_all(NBP, 7)
        lattice = NBP.initdat(N)
        for _ in range(10):
            NBP.MC_step(lattice, 0.5, N)
        results.append(lattice)
    numba.set_num_threads(threads)
    assert np.allclose(results[0], results[1])
    
    
#Physics Check
def equilibrium_averages(module, temp, seed):
    """Run 300 MC steps and average energy per cell and acceptance over the last 150."""
    if module is LL:
        np.random.seed(seed)
    else:
        seed_all(module, seed)
    lattice = module.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = module.MC_step(lattice, temp, N)
        if step >= 150:                                   # skip the equilibration part
            energies.append(NB.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    return np.mean(energies), np.mean(ratios)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original(module, temp):
    e_orig, r_orig = equilibrium_averages(LL, temp, seed=1)
    e_nb, r_nb = equilibrium_averages(module, temp, seed=1)
    assert e_nb == pytest.approx(e_orig, abs=0.15)     # energy per cell agrees
    assert r_nb == pytest.approx(r_orig, abs=0.03)     # acceptance ratio agrees

#Whole Program
def test_serial_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    NB.main("test", 5, N, 0.5, 0)
    files = list((tmp_path / "Outputs" / "Numba").glob("*.txt"))
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6, 4)
    assert "compile" in capsys.readouterr().out
    
def test_parallel_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    NBP.main("test", 5, N, 0.5, 0, 1)
    files = list((tmp_path / "Outputs" / "NumbaParallel").glob("*.txt"))
    assert len(files) == 1
    assert np.loadtxt(files[0]).shape == (6, 4)
    assert "Threads: 1" in capsys.readouterr().out
    
def test_parallel_command_line(tmp_path):
    import subprocess, sys, pathlib
    program = pathlib.Path(NBP.__file__)
    result = subprocess.run([sys.executable, str(program), "5", "20", "0.5", "0", "2"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert "Threads: 2" in result.stdout