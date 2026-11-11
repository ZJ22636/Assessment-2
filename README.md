# Accelerating Lebwohl-Lasher Model
Accelerating the provided 2D Lebwohl-Lasher Monte Carlo code (`LebwohlLasher.py`)
using NumPy, Numba, Cython and mpi4py each as a seperate program. A final optimised
version combines MPI with Cython as well as an improved method for calculating 
energy. Every version is tested against the original and timed with the same script.
All report figures are made in `Analysis.ipynb`.

Repository: https://github.com/ZJ22636/assessment-2

---

## Versions

| Program | Description |
|---|---|
| `LebwohlLasher.py` | Original code provided with bug fixes only |
| `NumPy_LebwohlLasher.py` | NumPy vectorisation with checkerboard update implemented |
| `Numba_LebwohlLasher.py` | Numba version serial |
| `NumbaParallel_LebwohlLasher.py` | Numbda parallel version with prange threads. |
| `Cython_LebwohlLasher.py` + `Cython_LL.pyx` | Cython with typed memoryviews and C maths serial |
| `CythonParallel_LebwohlLasher.py` + `CythonParallel_LL.pyx` | Cython with OpenMP threads |
| `MPI_LebwohlLasher.py` | mpi4py row strips with halo exchange. |
| `Optimised_LebwohlLasher.py` + `Optimised_LL.pyx` | MPI with Cython kernels and a cos 2theta / sin 2theta energy form. |

---

## Repo Layout

```
├── *_LebwohlLasher.py     the eight programs
├── *_LL.pyx               Cython kernels
├── setup_Cython.py        builds the Cython modules
├── timing.py              times any version over several lattice sizes
├── Analysis.ipynb         makes every figure in the report
├── tests/                 pytest tests, one file per version
└── Results/
    ├── Timings/           timing CSVs and machine_info.txt
    ├── Figures/           report figures
    ├── Physics_Check/     validation runs for every version
    └── Profiling/         cProfile output for the original code
```

Raw run files (`Outputs/`) and Cython build are git ignored.

---

## Requirements

Windows 11, 10 physcial cores, 12 threads

- Python 3.9.13
- NumPy 2.0.2
- Numba 0.60.0
- Cython 3.3.0
- mpi4py 4.1.2
- SciPy 1.13.1
- matplotlib 3.9.4
- pandas 2.3.3
- pytest 8.4.2
- psutil 7.2.2
- MS-MPI (for `mpiexec`)
- Microsoft C++ Build Tools (C compiler with OpenMP, for Cython)

---

## Setup & Building

Cython modules must be compiled before their programs tests or timings can be run.
From base folder:

```
python setup_Cython.py build_ext -fi
```

This builds `Cython_LL`, `CythonParallel_LL` and `Optimised_LL`.
Rebuild after changing any of the `.pyx` files.
Numba compiles itself so no building needed.

Run the tests with `python -m pytest`

Note: MPI tests are skipped if `mpiexec` isn't installed.

---

## Running Programs

```
python <PROGRAM> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [THREADS]
```

Commands used for report results:

```
python LebwohlLasher.py 50 200 0.5 0
python NumPy_LebwohlLasher.py 50 200 0.5 0
python Numba_LebwohlLasher.py 50 200 0.5 0
python NumbaParallel_LebwohlLasher.py 50 200 0.5 0 12
python Cython_LebwohlLasher.py 50 200 0.5 0
python CythonParallel_LebwohlLasher.py 50 200 0.5 0 12
mpiexec -n 10 python MPI_LebwohlLasher.py 50 200 0.5 0
mpiexec -n 10 python Optimised_LebwohlLasher.py 50 200 0.5 0
```

Physics checks used same settings for every version (MPI version below).

```
mpiexec -n 4 python Optimised_LebwohlLasher.py 5000 20 0.65 0
```

---

## Timing

```
python timing.py <PROGRAM> <LABEL> [THREADS or pPROCESSES]
```

Runs the program 3 times at each lattice size, currently set to 400, 800, for 
50 MCS at T* = 0.5 and saves results to `Results/Timings/<LABEL>.csv`.
For large lattice runs the sizes list in `timing.py` needs to be changed.

---

## Notes

- All timings were done with laptop plugged in as battery mode slows CPU down and stops acces to all cores.
- Laptop has 2 fast cores and 8 slow ones which effects scaling.
- A repeat of original code at size = 800 is left out of analysis and results
  because laptop went to sleep during that run.
