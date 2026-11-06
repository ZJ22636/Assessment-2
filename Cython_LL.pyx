"""
Cython version of Lebwohl-Lasher, C type version
variables now C types with cdef and lattice passed as 
typed memoryview
"""
import numpy as np
cimport cython
#=======================================================================
@cython.boundscheck(False) #skip checking indices inside array
@cython.wraparound(False) #no negative indices
cpdef double one_energy(double[:, ::1] arr, int ix, int iy, int nmax):
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
    cdef int ixm = (ix-1)%nmax # of the neighbours
    cdef int iyp = (iy+1)%nmax # with wraparound
    cdef int iym = (iy-1)%nmax #
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

#=======================================================================
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
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
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
    cdef double[:, ::1] Qab = np.zeros((3,3))
    cdef double[:, ::1] delta = np.eye(3,3)
    cdef int a, b, i, j
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    arr_np = np.asarray(arr) #numpy array
    cdef double[:, :, ::1] lab = np.vstack((np.cos(arr_np),np.sin(arr_np),np.zeros_like(arr_np))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Q = np.asarray(Qab)/(2*nmax*nmax)
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
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    cdef double scale=0.1+Ts
    cdef int accept = 0
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    cdef Py_ssize_t[:, ::1] xran = np.random.randint(0,high=nmax, size=(nmax,nmax)).astype(np.intp)
    cdef Py_ssize_t[:, ::1] yran = np.random.randint(0,high=nmax, size=(nmax,nmax)).astype(np.intp)
    cdef double[:, ::1] aran = np.random.normal(scale=scale, size=(nmax,nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)