import numpy as np
from numpy import random as rng
from proglearn.transformers import *
from split import BaseObliqueSplitter as BOS
from time import time

from numpy.testing import assert_almost_equal

def numpy_Q(X, y, spl):

    n, m = X.shape[0], X.shape[1]

    Q = np.zeros((n, m))
    
    # Impurity
    unique, count = np.unique(y, return_counts=True)
    count = count / n
    gini = 1 - np.sum(np.power(count, 2))
    Q[0, :] = gini
    Q[-1, :] = gini

    for j in range(m):

        idx = np.argsort(X[:, j])
        y_sort = y[idx]

        Q[1:-1, j] = np.array([spl.score(y_sort, i) for i in range(1, n - 1)])

    return Q

def Cython_Q(X, y):
    B = BOS()
    Q = B.score_matrix(X, y)
    return Q


rng.seed(0)
n = 1000
m = 20

numpy_time = 0
cython_time = 0
trials = 10
for i in range(trials):
    X = rng.rand(n, m)
    y = rng.randint(20, size=n)
    y = np.array(y, dtype=np.float64)

    print(i)
    spl = ObliqueSplitter(X, y, m, 0.5, 0)
    t1 = time()
    Q_numpy = numpy_Q(X, y, spl)
    t2 = time()
    numpy_time += (t2 - t1)

    t1 = time()
    Q_cython = Cython_Q(X, y)
    t2 = time()
    cython_time += (t2 - t1)

    assert_almost_equal(Q_numpy, Q_cython)

print("Numpy time: ", numpy_time/trials)
print("Cython time:", cython_time/trials)

