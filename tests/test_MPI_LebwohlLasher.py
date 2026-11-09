import numpy as np 
import pytest
from mpi4py import MPI
import LebwohlLasher as LL
import MPI_LebwohlLasher as MP
import subprocess
import sys
import pathlib
import shutil

N= 20
comm = MPI.COMM_WORLD

def single_strip(lattice):
    strip = np.zeros((lattice.shape[0] + 2, lattice.shape[1] ))
    strip[1:-1] = lattice
    return strip

#checks with 1 rank
def test_strip_energies_match_one_energy():
    np.random.seed(3)
    lattice = LL.initdat(N)
    strip = single_strip(lattice)
    MP.halo_exchange(strip, comm, 0, 0)
    original = np.array([[LL.one_energy(lattice, i, j, N) for j in range(N)] for i in range(N)])
    assert MP.strip_energy(strip) == pytest.approx(original)
    
def test_low_temperature_orders():
    np.random.seed(0)
    strip = single_strip(LL.initdat(10))
    rng = np.random.default_rng(0)
    start_energy = MP.all_energy(strip, comm, 0, 0)
    for i in range(100):
        MP.MC_step(strip, 0.1, 0, rng, comm, 0,0)
    assert MP.all_energy(strip, comm, 0,0) < start_energy
    assert MP.get_order(strip, 10, comm) > 0.8
    
#Multi rank Checks 
PROGRAM = pathlib.Path(MP.__file__)
needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None, reason= "mpiexec not found")

def run_mpi(processes, steps, size, temp, seed, folder):
    result = subprocess.run(["mpiexec", "-n", str(processes), sys.executable, str(PROGRAM),
                             str(steps), str(size), str(temp), "0", str(seed)],
                            capture_output=True, text=True, cwd=folder, timeout=300)
    files = list ((folder / "Outputs" / "MPI").glob("*.txt"))
    data = np.loadtxt(files[0]) if files else None
    return result, data

@needs_mpiexec
@pytest.mark.parametrize("processes", [1, 2,3, 4])
def test_start_matches_original_any(processes, tmp_path):
    result, data = run_mpi(processes, 1, N, 0.5, 11, tmp_path)
    assert f"Processes: {processes}" in result.stdout
    np.random.seed(11)                                   
    lattice = LL.initdat(N)
    assert data[0, 2] == pytest.approx(LL.all_energy(lattice, N), abs=1e-3)
    assert data[0, 3] == pytest.approx(LL.get_order(lattice, N).real, abs=1e-3) 
    

@needs_mpiexec
def test_odd_size_rejected(tmp_path):
    result, data = run_mpi(2,1,21,0.5,1,tmp_path)
    assert result.returncode != 0
    assert data is None
    


def original_equilibrium(temp, seed):
    np.random.seed(seed)
    lattice = LL.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = LL.MC_step(lattice, temp, N)
        if step >= 150:
            energies.append(LL.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    return np.mean(energies), np.mean(ratios)


@needs_mpiexec
@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original_4_processes(temp, tmp_path):
    result, data = run_mpi(4, 300, N, temp, 1, tmp_path)
    e_mpi = np.mean(data[151:, 2]) / N**2         
    r_mpi = np.mean(data[151:, 1])
    e_orig, r_orig = original_equilibrium(temp, 1)
    assert e_mpi == pytest.approx(e_orig, abs=0.15)
    assert r_mpi == pytest.approx(r_orig, abs=0.03)