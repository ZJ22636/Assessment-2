"""
compiled kernel of best LebwohlLAsher
each MPI process calls on its own strip of rows,
with halo row above and below. so MPI version of numpy update and sums
rewritten as typed cython loops with C maths
Maths update: instead of angle theta kernels use c = cos(2*theta)
and s = sin(2*theta) for every cell, stored together in cs[row, column, 0 or 1]

"""
cimport cython
from libc.math cimport cos, sin, exp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mc_half_step(double[:,::1] theta, double[:,:,::1] cs, double[:,::1] aran,
                 double[:,::1] rran, int colour, int row0, double Ts):
    #one half step of checkerboard update on this strip
    cdef int rows = theta.shape[0]
    cdef int nmax = theta.shape[1]
    cdef int i, ii, j, jl, jr, accept = 0
    cdef double C, S, new, cn, sn, dE
    for i in range(rows):
        ii = i + 1 #row i ins cs, cs has halo ring on top
        for j in range((colour + row0 + i) % 2, nmax, 2):
            jl = j - 1 if j > 0 else nmax - 1
            jr = j + 1 if j < nmax - 1 else 0 
            C = cs[ii-1, j, 0] + cs[ii+1, j, 0] + cs[ii, jl, 0] + cs[ii, jr, 0]   #sum of neighbours
            S = cs[ii-1, j, 1] + cs[ii+1, j, 1] + cs[ii, jl, 1] + cs[ii, jr, 1] #sum of neighoburs
            new = theta[i,j] + aran[i,j]
            cn = cos(2.0*new)
            sn = sin(2.0*new)
            dE = -0.75*((cn - cs[ii, j, 0])*C + (sn - cs[ii, j, 1])*S) #energy change of 4 bonds
            if dE <= 0.0 or exp(-dE/Ts) >= rran[i,j]: #metropolis test
                theta[i,j] = new
                cs[ii, j, 0] = cn
                cs[ii, j, 1] = sn
                accept += 1
    return accept

@cython.boundscheck(False)
@cython.wraparound(False)
def local_energy(double[:, :, ::1] cs):
    #sum of energies of ranks cells, halos must be up to date
    cdef int rows = cs.shape[0] -2
    cdef int nmax = cs.shape[1]
    cdef int i, j, jl, jr
    cdef double C, S, en = 0.0
    for i in range(1, rows+1):
        for j in range(nmax):
            jl = j - 1 if j > 0 else nmax - 1
            jr = j + 1 if j < nmax - 1 else 0
            C = cs[i-1, j, 0] + cs[i+1, j, 0] + cs[i, jl, 0] + cs[i, jr, 0]
            S = cs[i-1, j, 1] + cs[i+1, j, 1] + cs[i, jl, 1] + cs[i, jr, 1]
            en += -1.0 - 0.75*(cs[i, j, 0]*C + cs[i, j, 1]*S) 
    return en
@cython.boundscheck(False)
@cython.wraparound(False)
def order_sums(double[:, :, ::1] cs):
    #sum of cos x cos sin x sin and cos x sin over ranks cells
    cdef int rows = cs.shape[0] - 2
    cdef int nmax = cs.shape[1]
    cdef int i, j
    cdef double sc = 0.0, ss = 0.0
    for i in range(1, rows+1):
        for j in range(nmax):
            sc += cs[i, j, 0]
            ss += cs[i, j, 1]
    return sc, ss





                  

