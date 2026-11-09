import subprocess
import sys
import shutil
import pathlib
import numpy as np
import pytest
import LebwohlLasher as LL
import MPI_LebwohlLasher as MP
import Optimised_LebwohlLasher as OP 

N = 20
needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None, reason = "mpiexec not found")

def run_mpi(program, folder_name, processes, steps, size, temp, seed, folder):
    result = subprocess.run(["mpiexec", "-n", str(processes), sys.executable, str(program),
                             str(steps), str(size), str(temp), "0", str(seed)],
                            capture_output=True, text=True, cwd=folder, timeout=300)
    files = list((folder / "Outputs" / folder_name).glob("*.txt"))
    data = np.loadtxt(files[0]) if files else None
    return result, data

@needs_mpiexec
@pytest.mark.parametrize("processes", [1, 2, 3, 4])
def test_identical_to_MPI_version(processes, tmp_path):
    emp, mpi = run_mpi(pathlib.Path(MP.__file__), "MPI", processes, 20, N, 0.5, 7, tmp_path)
    emp, opt = run_mpi(pathlib.Path(OP.__file__), "Optimised", processes, 20, N, 0.5, 7, tmp_path)
    assert opt == pytest.approx(mpi, abs=1e-3)