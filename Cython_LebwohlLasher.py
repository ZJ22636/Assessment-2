"""
Cython version of Lebwohl Lasher code

simulation functions are in Cython_LL.pyx and compiled
by setup_Cython.py. this file is for python parts

Build:
python setup_Cython.py build_ext -fi

Run: 
python Cython_LebwholLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
"""

import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from Cython_LL import one_energy, all_energy, get_order, MC_step

VERSION = "Cython" #label for output folders and headers

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
def main(program, nsteps, nmax, temp, pflag):
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
    # Create and initialise lattice
    t_start = time.perf_counter()
    lattice = initdat(nmax)
    t_setup = time.perf_counter() - t_start
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    #changed np.dtype to np.float64 to avoid outdated numpy error
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    t_start = time.perf_counter()
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)
    t_setup += time.perf_counter() - t_start

    # Begin doing and timing some MC steps.
    t_mc = t_energy = t_order = 0.0
    initial = time.perf_counter()
    for it in range(1,nsteps+1):
        t0 = time.perf_counter()
        ratio[it] = MC_step(lattice,temp,nmax)
        
        t1 = time.perf_counter()
        energy[it] = all_energy(lattice,nmax)
        
        t2 = time.perf_counter()
        order[it] = get_order(lattice,nmax)
        
        t3 = time.perf_counter()
        t_mc += t1 -t0
        t_energy += t2 - t1
        t_order += t3 - t2
    final = time.perf_counter()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps],runtime))
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
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
