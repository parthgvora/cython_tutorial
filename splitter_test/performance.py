from proglearn.transformers import ObliqueTreeClassifier as OTC
import numpy as np
from time import time

N = [100, 200, 400, 800, 1600, 3200]
d = 10
reps = 5

print(len(N), reps)

for n in N:

    for r in range(reps):

        X = np.random.rand(n, d)
        y = np.random.randint(20, size=n)
        clf = OTC(random_state=0)

        start_time = time()
        clf.fit(X, y)
        end_time = time()

        print(n, end_time - start_time)


