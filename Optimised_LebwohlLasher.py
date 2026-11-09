"""
Supposedly fastest version of 2D Lebwohl Lasher code.
MPI processes each running compiled Cython kernel on own strip of rows.

Lattice is split into horizontal strips one per rank.
Halo rows used to swap between neighbours.
Work of each rank done by compiled loops instead of numpy,
using cos(2theta) and sin(2theta) so that energies need no trig and each move 
needs only two trig calls.

Build:
python setup_Cython.py build_ext -fi

Run:
mpiexec -n <PROCESSES> python Optimised_LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEED]

SIZE must be even for checkerboard
"""

import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from Optimised_LL import mc_half_step, local_energy, order_sums 

VERSION = "Optimised" #label for output folders and headers

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    
    #changed output filename as date and time was annoying
    folder = os.path.join("Outputs", VERSION)
    os.makedirs(folder, exist_ok=True)
    base = "MCS{:d}_T{:.2f}_Size{:d}".format(nsteps, Ts, nmax)
    v = 1
    while os.path.exists(os.path.join(folder, "{}_V{:d}.txt".format(base, v))):
        v += 1
    filename = os.path.join(folder, "{}_V{:d}.txt".format(base, v))
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Version:             {:s}".format(VERSION),file=FileOut) #which code made the file
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en

def halo_exchange(strip, comm, up, down):
    #send and recieve halo rows from and into nieghbours with wraparound for bottom and top
    comm.Sendrecv(strip[1], dest = up, recvbuf=strip[-1], source= down) #first row goes up bottom halo comes from below
    comm.Sendrecv(strip[-2], dest=down, recvbuf=strip[0], source = up) # last row goes down, top halo comes from above
    

#=======================================================================
def all_energy(cs, comm, up, down):
    """
    energy of whole lattice, each rank adds its own rows 
    then allreduce adds all together
    """
    halo_exchange(cs, comm, up, down) #current halos
    local = np.array([local_energy(cs)])
    total = np.zeros(1)
    comm.Allreduce(local, total, op = MPI.SUM) #add up totals
    return total[0]
    
#=======================================================================
def get_order(cs, nmax, comm):
    """
    order parameter of lattice using Q tensor
    ranks sum own rows allreduce combines
    """
    local = np.array(order_sums(cs))
    sums = np.zeros(2)
    comm.Allreduce(local, sums, op = MPI.SUM)
    mean_c, mean_s = sums/(nmax*nmax)
    return 0.25 + 0.75*np.sqrt(mean_c**2 + mean_s**2) #workout eignevalue of Q by hand
#=======================================================================
def MC_step(theta, cs, Ts, row0, rng, comm, up,down):
    """
    One MCS using checkerboard update, update black cells
    then white cells. Before each half step halo rows are
    refreshed so cells on edge of strip see neighbours most up 
    to date values. each rank updates own rows at once.every cell tried once per step
    one array of random changes and one of random numbers cover both colours
    """
    
    scale=0.1+Ts
    rows, nmax = theta.shape
    aran = rng.normal(scale=scale, size=(rows,nmax))
    rran = rng.random(size=(rows,nmax))
    accept = 0
    for colour in (0,1): #black cells the white
        halo_exchange(cs, comm, up, down) #latest edge row of neighbour
        accept += mc_half_step(theta, cs, aran, rran, colour, row0, Ts)
    total = comm.allreduce(int(accept), op = MPI.SUM)
      
    return total/(nmax*nmax)
        
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, seed=None):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if nmax % 2 !=0 or size > nmax:
        raise ValueError("SIZE must be even and at least number of ranks")
    counts = [nmax//size + (1 if r < nmax % size else 0) for r in range(size)] # rows per process
    rows = counts[rank] #rows per rank
    row0 = sum(counts[:rank]) #glob number of ranks first row
    sizes = [c*nmax for c in counts]
    starts = [sum(sizes[:r]) for r in range(size)]
    up = (rank-1) % size
    down = (rank+1) % size
    
    #each rank gets own random number from shared seed
    if seed is None:
        seed = np.random.SeedSequence().entropy % 2**32 #new random seed each run
    seed = comm.bcast(seed, root = 0)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(size)[rank])
        
    # Create and initialise lattice
    t_start = time.perf_counter()
    lattice = None
    if rank == 0:
        np.random.seed(seed)
        lattice = initdat(nmax)
        plotdat(lattice,pflag,nmax)
    theta = np.zeros((rows, nmax))
    comm.Scatterv([lattice, sizes, starts, MPI.DOUBLE] if rank == 0 else None,
                  theta, root=0) # each rank gets its rows
    cs = np.zeros((rows+2,nmax,2)) #cos(2theta), sin(2theta), plus halo rows
    cs[1:-1,:,0] = np.cos(2*theta)
    cs[1:-1,:,1] = np.sin(2*theta)
    t_setup = time.perf_counter() - t_start
    # Create arrays to store energy, acceptance ratio and order parameter
    #changed np.dtype to np.float64 to avoid outdated numpy error
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    t_start = time.perf_counter()
    energy[0] = all_energy(cs,comm,up,down)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(cs,nmax,comm)
    t_setup += time.perf_counter() - t_start

    # Begin doing and timing some MC steps.
    comm.Barrier()
    t_mc = t_energy = t_order = 0.0
    initial = time.perf_counter()
    for it in range(1,nsteps+1):
        t0 = time.perf_counter()
        ratio[it] = MC_step(theta, cs, temp, row0, rng,comm,up,down)
        
        t1 = time.perf_counter()
        energy[it] = all_energy(cs,comm,up,down)
        
        t2 = time.perf_counter()
        order[it] = get_order(cs,nmax,comm)
        
        t3 = time.perf_counter()
        t_mc += t1 -t0
        t_energy += t2 - t1
        t_order += t3 - t2
    final = time.perf_counter()
    runtime = final-initial
    
    #get rows back on rank 0
    if rank == 0:
        lattice = np.zeros((nmax, nmax))
    comm.Gatherv(theta, [lattice,sizes, starts, MPI.DOUBLE]if rank == 0 else None, root=0)
    
    # Final outputs
    if rank == 0:
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}, Processes: {:d}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp, size, order[nsteps],runtime))
        # Plot final frame of lattice and generate output file
        t_start = time.perf_counter()
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        t_save = time.perf_counter() - t_start
        print("Timings: setup: {:8.6f} s, MC_step: {:8.6f} s, all_energy: {:8.6f} s, get_order: {:8.6f} s, savedat: {:8.6f} s".format(t_setup, t_mc, t_energy, t_order, t_save))
        plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) in (5,6):
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        SEED = int(sys.argv[5]) if len(sys.argv) == 6 else None
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, SEED)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEED]".format(sys.argv[0]))
#=======================================================================
