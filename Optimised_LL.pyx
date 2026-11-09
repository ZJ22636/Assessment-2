"""
compiled kernel of best LebwohlLAsher
each MPI process calls on its own strip of rows,
with halo row above and below. so MPI version of numpy update and sums
rewritten as typed cython loops with C maths
"""
cimport cython
from libc.math cimport cos, sin, exp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double bond(double a, double b) noexcept nogil:
    cdef double c = cos(a - b)
    return 0.5*(1.0 - 3.0*c*c)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mc_half_step(double[:, ::1] strip, double[:, ::1] aran, double[:, ::1] rran,
                 int colour, int row0, double Ts):
    #one half step of checkerboard update on this strip
    cdef int rows = strip.shape[0] - 2
    cdef int nmax = strip.shape[1]
    cdef int i, j, jl, jr, accept = 0
    cdef double old, new, up, down, left, right, en0, en1
    for i in range(1, rows+1): #so own rows not halo
        for j in range((colour + row0 + i - 1) % 2, nmax, 2): #cells of colour
            jl = j -1 if j> 0 else nmax - 1
            jr = j + 1 if j < nmax -1 else 0
            up = strip[i-1,j]
            down = strip[i+1, j]
            left = strip[i, jl]
            right = strip[i, jr]
            old = strip[i, j]
            new = old + aran[i-1, j]
            en0 = bond(old, up) + bond(old, down) + bond(old, left) + bond(old, right)
            en1 = bond(new, up) + bond(new, down) + bond(new, left) + bond(new, right)
            if en1 <= en0 or exp(-(en1- en0)/Ts) >= rran[i-1, j]:
                strip[i,j] = new
                accept += 1
    return accept

@cython.boundscheck(False)
@cython.wraparound(False)
def local_energy(double[:,::1] strip):
    #sum of energies of ranks cells, halos must be up to date
    cdef int rows = strip.shape[0] -2
    cdef int nmax = strip.shape[1]
    cdef int i, j, jl, jr
    cdef double t, en = 0.0
    for i in range(1, rows+1):
        for j in range(nmax):
            jl = j - 1 if j > 0 else nmax - 1
            jr = j + 1 if j < nmax -1 else 0
            t = strip[i, j]
            en += bond(t, strip[i-1, j]) + bond(t, strip[i+1, j]) + bond(t, strip[i, jl]) + bond(t, strip[i, jr])
    return en

@cython.boundscheck(False)
@cython.wraparound(False)
def order_sums(double[:, ::1] strip):
    #sum of cos x cos sin x sin and cos x sin over ranks cells
    cdef int rows = strip.shape[0] -2
    cdef int nmax = strip.shape[1]
    cdef int i, j
    cdef double c, s, sxx = 0.0, syy = 0.0, sxy = 0.0
    for i in range(1, rows+1):
        for j in range(nmax):
            c = cos(strip[i, j])
            s = sin(strip[i,j])
            sxx += c*c
            syy += s*s
            sxy += c*s
    return sxx, syy, sxy





                  

