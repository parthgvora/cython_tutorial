cimport cython

import numpy as np
DTYPE = np.float64

#from libc.stdlib cimport qsort

"""
cdef int cmpf(const void* a, const void* b) nogil:
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1

"""

# Computes the gini score for a split
# 0 < t < len(y)
@cython.boundscheck(False) #Deactivate bounds checking
@cython.wraparound(False) #Deactivate negative indexing
cpdef double score(double[:] y, int t):

    cdef double length = y.shape[0]
    cdef double left_gini = 1.0
    cdef double right_gini = 1.0
    cdef double gini = 0
    cdef int i = 0
    cdef double temp = 0.0
    cdef double count = 0.0
    
    cdef double[:] left = y[:t]
    cdef double[:] right = y[t:]

    cdef int l_length = left.shape[0]
    cdef int r_length = right.shape[0]

    # Sort the arrays
    left = np.sort(left)
    right = np.sort(right)

    # Slower and wrong???
    #qsort(&left[0], left.shape[0], left.strides[0], &cmpf)
    #qsort(&right[0], right.shape[0], right.strides[0], &cmpf)

    # Count unique elements
    # And compute gini index
    temp = left[0]
    count = 1.0
    for i in range(1, l_length):
        if left[i] != temp:
            count = count / l_length
            count = count * count
            left_gini = left_gini - count

            count = 1.0
            temp = left[i]

        else:
            count = count + 1

    count = count / l_length
    count = count * count
    left_gini = left_gini - count

    temp = right[0]
    count = 1.0
    for i in range(1, r_length):
        if right[i] != temp:
            count = count / r_length
            count = count * count
            right_gini = right_gini - count

            count = 1.0
            temp = right[i]

        else:
            count = count + 1

    count = count / r_length
    count = count * count
    right_gini = right_gini - count

    gini = (l_length / length) * left_gini + (r_length / length) * right_gini
    return gini

@cython.boundscheck(False) #Deactivate bounds checking
@cython.wraparound(False) #Deactivate negative indexing
cpdef double[:, :] score_matrix(double[:, :] X, double[:] y, double node_impurity,
        int n_samples, int proj_dims):

    cdef int i = 0
    cdef int j = 0

    Q = np.zeros((n_samples, proj_dims), dtype=DTYPE)
    cdef double[:, :] Q_view = Q

    idx = np.zeros(n_samples, dtype=np.int)
    cdef int[:] idx_view = idx

    y_sort = np.zeros(n_samples, dtype=DTYPE)
    cdef double[:] y_sort_view = y_sort

    # No split = just impurity of the whole thing
    Q_view[0, :] = node_impurity
    Q_view[n_samples - 1, :] = node_impurity

    for j in range(0, proj_dims):
        
        idx_view = np.argsort(X[:, j])
        
        for i in range(0, n_samples):
            y_sort_view[i] = y[idx_view[i]]

        for i in range(1, n_samples - 1):
            Q_view[i, j] = score(y_sort_view, i)

    return Q



