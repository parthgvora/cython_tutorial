cimport cython

import numpy as np

from libcpp.unordered_map cimport unordered_map
from cython.operator import dereference, postincrement

from cython.parallel import prange

# Computes the gini score for a split
# 0 < t < len(y)
@cython.boundscheck(False) #Deactivate bounds checking
@cython.wraparound(False) #Deactivate negative indexing
cdef double score(double[:] y, int t) nogil:
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
    cdef double l_len_doub = left.shape[0]
    cdef double r_len_doub = right.shape[0]

    cdef unordered_map[double, double] count_left
    cdef unordered_map[double, double] count_right

    cdef unordered_map[double, double].iterator left_it = count_left.begin()
    cdef unordered_map[double, double].iterator right_it = count_right.begin()

    #print([y[i] for i in range(y.shape[0])])

    # Count all unique elements, store in hashmap
    for i in range(0, l_length):
        temp = left[i]
        count_left[temp] += 1

    for i in range(0, r_length):
        temp = right[i]
        count_right[temp] += 1
   
    # Compute left gini and right gini
    left_it = count_left.begin()
    while (left_it != count_left.end()):
        count = dereference(left_it).second
        count = count / l_len_doub
        count = count * count
        left_gini -= count
        postincrement(left_it)

    right_it = count_right.begin()
    while (right_it != count_right.end()):
        count = dereference(right_it).second
        count = count / r_len_doub
        count = count * count
        right_gini -= count
        postincrement(right_it)

    gini = (l_len_doub / length) * left_gini + (r_len_doub / length) * right_gini
    return gini

@cython.boundscheck(False) #Deactivate bounds checking
@cython.wraparound(False) #Deactivate negative indexing
cpdef score_matrix(double[:, :] X, double[:] y, double node_impurity):

    cdef int n_samples = X.shape[0]
    cdef int proj_dims = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    cdef long temp_int = 0;

    Q = np.zeros((n_samples, proj_dims), dtype=np.float64)
    cdef double[:, :] Q_view = Q

    idx = np.zeros(n_samples, dtype=np.int)
    cdef long[:] idx_view = idx

    y_sort = np.zeros(n_samples, dtype=np.float64)
    cdef double[:] y_sort_view = y_sort

    # No split = just impurity of the whole thing
    Q_view[0, :] = node_impurity
    Q_view[n_samples - 1, :] = node_impurity

    for j in range(0, proj_dims):
       
        # Correct so far
        idx_view = np.argsort(X[:, j])
        for i in range(0, n_samples):
            temp_int = idx_view[i]
            y_sort_view[i] = y[temp_int]

        for i in prange(1, n_samples - 1, nogil=True):
            Q_view[i, j] = score(y_sort_view, i)

    return Q



