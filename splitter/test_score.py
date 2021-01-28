import numpy as np
from numpy import random as rng
from proglearn.transformers import *
from split import score 

from time import time
random_state = 0
rng.seed(random_state)

density = 0.5
proj_dims = 5
"""
X = rng.rand(10, 10)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)

spl1 = ObliqueSplitter(X, y, proj_dims, density, random_state)

for i in range(1, 10):
    score1 = spl1.score(y, i)
    score2 = score(y, i, 10)

    print(score1, score2)

"""
n = 10000
spos = 5000
trials = 10

sporf_time = 0
cython_time = 0
for i in range(trials):
    X = rng.rand(n, n)
    y = rng.randint(spos, size=n)
    y = np.array(y, dtype=np.float64)

    spl1 = ObliqueSplitter(X, y, proj_dims, density, random_state)
    t1 = time()
    score1 = spl1.score(y, spos)
    t2 = time()
    sporf_time += (t2 - t1)

    t1 = time()
    score2 = score(y, spos)
    t2 = time()
    cython_time += (t2 - t1)

    print("SPORF  score: ", score1)
    print("Cython score: ", score2)

print("SPORF time :", sporf_time/trials)
print("Cython time:", cython_time/trials)
