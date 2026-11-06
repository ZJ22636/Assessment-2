"""
Parallel Cython version of Lebwohl-Lasher
Uses OpenMP, GIL released so shares work between
COU threads
"""
import numpy as np
cimport cython
from libc.math cimport cos, sin, exp
cimport openmp
from cython.parallel cimport prange

#=======================================================================
def set_threads(int threads):
    openmp.omp_set_num_threads(threads)

@cython.boundscheck(False) #skip checking indices inside array
@cython.wraparound(False) #no negative indices
@cython.cdivision(True) #C style division
cpdef double one_energy(double[:, ::1] arr, int ix, int iy, int nmax) noexcept nogil:
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
    cdef double en = 0.0
    cdef double ang
    cdef int ixp = (ix+1)%nmax # These are the coordinates
    cdef int ixm = (ix-1+nmax)%nmax # of the neighbours
    cdef int iyp = (iy+1)%nmax # with wraparound
    cdef int iym = (iy-1+nmax)%nmax 
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    return en

#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double all_energy(double[:, ::1] arr, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef double enall = 0.0
    cdef int i, j
    for i in prange(nmax, nogil=True): #rows now shared between threads
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax) #adds threads totals
    return enall

#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def get_order(double[:, ::1] arr, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    cdef double sxx = 0.0, syy = 0.0, sxy = 0.0
    cdef double c, s
    cdef int i,j

    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            c = cos(arr[i, j])
            s = sin(arr[i, j])
            sxx += c*c
            syy += s*s
            sxy += c*s
    cdef double n2 = nmax*nmax
    Qab = np.array([[3*sxx - n2, 3*sxy, 0.0],
                    [3*sxy, 3*syy - n2, 0.0],
                    [0.0, 0.0, -n2]])

    Q = Qab/(2*n2)
    eigenvalues,eigenvectors = np.linalg.eigh(Q)
    return eigenvalues.max()
#=======================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def MC_step(double[:, ::1] arr, double Ts, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    if nmax % 2 != 0:
        raise ValueError("Checkerboard needs an even lattice size")
    
    cdef double scale=0.1+Ts
    cdef int accept = 0
    cdef int colour, ix, iy
    cdef double ang, en0, en1, boltz
    cdef double[:, ::1] aran
    cdef double[:, ::1] rran
    
    for colour in range(2):
        aran = np.random.normal(scale=scale, size=(nmax,nmax))
        rran = np.random.uniform(0.0, 1.0, size=(nmax,nmax))
        for ix in prange(nmax, nogil=True):
            for iy in range((ix+colour)%2, nmax, 2):
                ang = aran[ix, iy]
                en0 = one_energy(arr,ix,iy,nmax)
                arr[ix,iy] += ang
                en1 = one_energy(arr, ix, iy,nmax)
                if en1<=en0:
                    accept += 1
                else:
                    boltz = exp( -(en1 - en0) / Ts)
                    if boltz >= rran[ix,iy]:
                        accept += 1
                    else: 
                        arr[ix,iy] -= ang

    return accept/(nmax*nmax)