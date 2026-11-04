import numpy as np
import pytest
import LebwohlLasher as LL

N = 20

#setting up lattice
def test_initdat_shape_and_range():
    lattice = LL.initdat(N)
    assert lattice.shape == (N, N) 
    assert lattice.min() >= 0 and lattice.max() < 2 * np.pi
    
#Energy
def test_aligned_cell_energy():
    lattice = np.zeros((N,N)) #every cell points same way
    assert LL.one_energy(lattice, 3, 4, N) == pytest.approx(-4.0)
    
def test_aligned_total_energy():
    lattice = np.zeros((N,N))
    assert LL.all_energy(lattice, N) == pytest.approx(-4.0 * N * N)
    
def test_perp_energy():
    i, j = np.indices((N,N))
    lattice = np.where((i, j) % 2 == 0, 0.0, np.pi /2) #checkerboard of 0 and 90 degrees
    assert LL.one_energy(lattice, 3, 4, N) == pytest.approx(2.0)
    
def test_periodic_boundaries():
    lattice = np.zeros((N,N))
    lattice[0,0] = np.pi / 2 #turn corner cells 90 degrees
    assert LL.one_energy(lattice, 0, 0, N) == pytest.approx(2.0)
    assert LL.one_energy(lattice, N - 1, 0, N) == pytest.approx(-2.5)
    
def test_head_tail_symmetry():
    np.random.seed(1)
    lattice = LL.initdat(N)
    flipped = lattice.copy()
    flipped[2, 2] += np.pi # theta and theta+pi are the same director
    assert LL.one_energy(flipped, 2, 2, N) == pytest.approx(LL.one_energy(lattice, 2, 2, N))
 
 
def test_rotation_invariance():
    np.random.seed(2)
    lattice = LL.initdat(N)
    rotated = lattice + 0.7  # rotate every cell by the same angle
    assert LL.all_energy(rotated, N) == pytest.approx(LL.all_energy(lattice, N))
    assert LL.get_order(rotated, N) == pytest.approx(LL.get_order(lattice, N))
 
 
#Order Parameter
 
def test_aligned_order():
    assert LL.get_order(np.zeros((N, N)), N) == pytest.approx(1.0)
 
 
def test_random_order():
    np.random.seed(0)
    lattice = LL.initdat(100)
    assert 0.2 < LL.get_order(lattice, 100) < 0.3   # about 0.25 for random angles in 2D
 
 
#MCS
 
def test_acceptance_ratio_range():
    np.random.seed(0)
    lattice = LL.initdat(N)
    ratio = LL.MC_step(lattice, 0.5, N)
    assert 0.0 <= ratio <= 1.0
 
 
def test_mc_step_reproducible():
    np.random.seed(42)
    a = LL.initdat(N)
    LL.MC_step(a, 0.5, N)
    np.random.seed(42)
    b = LL.initdat(N)
    LL.MC_step(b, 0.5, N)
    assert np.array_equal(a, b) # same seed gives exactly the same result
 
 
def test_low_temperature_orders():
    np.random.seed(0)
    lattice = LL.initdat(N)
    start_energy = LL.all_energy(lattice, N)
    for _ in range(100):
        LL.MC_step(lattice, 0.1, N)
    assert LL.all_energy(lattice, N) < start_energy
    assert LL.get_order(lattice, N) > 0.8
 
 
def test_high_temperature_stays_disordered():
    np.random.seed(0)
    lattice = LL.initdat(20)
    for _ in range(100):
        LL.MC_step(lattice, 2.0, 20)
    assert LL.get_order(lattice, 20) < 0.5
 
 
#Whole program 
 
def test_main_runs_and_saves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path) # write the output file into a temporary folder
    LL.main("test", 5, N, 0.5, 0)
    files = list(tmp_path.glob("LL-Output-*.txt"))
    assert len(files) == 1
    data = np.loadtxt(files[0])
    assert data.shape == (6, 4) # steps 0 to 5, four columns
    assert "Order" in capsys.readouterr().out
    
    