import subprocess
import sys
import shutil
import pathlib
import numpy as np
import pytest
import LebwohlLasher as LL
import MPI_LebwohlLasher as MP
import Optimised_LebwohlLasher as OP 
from mpi4py import MPI

N = 20
needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None, reason = "mpiexec not found")
PROGRAM =pathlib.Path(OP.__file__)
comm = MPI.COMM_WORLD

def run_mpi(processes, steps, temp, seed, folder):
    """Run the program with mpiexec and return its saved data."""
    subprocess.run(["mpiexec", "-n", str(processes), sys.executable, str(PROGRAM),
                    str(steps), str(N), str(temp), "0", str(seed)],
                   capture_output=True, text=True, cwd=folder, timeout=300)
    return np.loadtxt(next((folder / "Outputs" / "Optimised").glob("*.txt")))

def test_new_maths_matches_original():
    np.random.seed(4)
    lattice = LL.initdat(N)
    cs = np.zeros((N + 2, N, 2)) # cos(2theta), sin(2theta) with halo rows
    cs[1:-1, :, 0] = np.cos(2 * lattice)
    cs[1:-1, :, 1] = np.sin(2 * lattice)
    assert OP.all_energy(cs, comm, 0, 0) == pytest.approx(LL.all_energy(lattice, N))
    assert OP.get_order(cs, N, comm) == pytest.approx(LL.get_order(lattice, N).real)
    
@needs_mpiexec
@pytest.mark.parametrize("processes", [1, 4])
def test_start_matches_original(processes, tmp_path):
    data = run_mpi(processes, 1, 0.5, 11, tmp_path)
    np.random.seed(11)
    lattice = LL.initdat(N)
    assert data[0, 2] == pytest.approx(LL.all_energy(lattice, N), abs=1e-3)
    assert data[0, 3] == pytest.approx(LL.get_order(lattice, N).real, abs=1e-3)


@needs_mpiexec
@pytest.mark.parametrize("temp", [0.5, 1.5])
def test_same_equilibrium_as_original(temp, tmp_path):
    data = run_mpi(4, 300, temp, 1, tmp_path)
    np.random.seed(1)
    lattice = LL.initdat(N)
    energies, ratios = [], []
    for step in range(300):
        ratio = LL.MC_step(lattice, temp, N)
        if step >= 150:
            energies.append(LL.all_energy(lattice, N) / N**2)
            ratios.append(ratio)
    assert np.mean(data[151:, 2]) / N**2 == pytest.approx(np.mean(energies), abs=0.15)
    assert np.mean(data[151:, 1]) == pytest.approx(np.mean(ratios), abs=0.03)