import numpy as np 
import cython 
cimport numpy as cnp
from libc.math cimport isnan
cdef extern from "math.h":
    double isnan(double x)
    double fabs(double x)


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.ndarray[DTYPE_t, ndim=1] DTYPE_1D

ctypedef DTYPE_1D D_ARRAY_1D

def solve(field, D_ARRAY, x0, double T, double dt)
    cdef double t, h, half_h
    cdef int i, N = x0.shape[0]

    h = dt
    half_h = h / 2.0
    cdef DTYPE_1D t_points = np.arange(0, T, dtype = DTYPE_T)
    cdef int num_steps = t_points.shape[0]

    cdef cnp.ndarray result = np.zeroes((num_steps, N), dtype = DTYPE_T)

    cdef DTYPE_1D x = x0.copy()
    cdef DTYPE_1D k1 = np.empty(N, dtype = DTYPE_T)
    cdef DTYPE_1D k2 = np.empty(N, dtype = DTYPE_T)
    cdef DTYPE_1D k3 = np.empty(N, dtype = DTYPE_T)
    cdef DTYPE_1D k4 = np.empty(N, dtype = DTYPE_T)
    cdef DTYPE_1D result = np.empty(N, dtype = DTYPE_T)
    cdef int i

    for i in range(num_steps):
        t = t_points[i]

        result[i, :] = x

        k1 = field(t, x) 
        
        x_temp[:] = x + half_h * k1
        k2 = field(t + half_h, x_temp)
        
        x_temp[:] = x + half_h * k2
        k3 = field(t + half_h, x_temp)
        
        x_temp[:] = x + h * k3
        k4 = field(t + h, x_temp)
        
        x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result