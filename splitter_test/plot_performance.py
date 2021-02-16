import numpy as np
import matplotlib.pyplot as plt


def read_times(fn):
    first = True
    
    for line in open(fn):

        line = line.split()
        if first:
            N = int(line[0])
            reps = int(line[1])
            X = np.zeros(N)
            y = np.zeros(N)
            i = 0
            first = False

        else:
            X[int(i / reps)] = int(line[0])
            y[int(i / reps)] += float(line[1]) / reps
            i += 1
    
    return X, y


cython_X, cython_y = read_times("cython_sporf.tsv")
python_X, python_y = read_times("python_sporf.tsv")

plt.figure()
plt.plot(cython_X, cython_y, label="Cython")
plt.plot(python_X, python_y, label="Python")
plt.xlabel("Number of Samples")
plt.ylabel("Mean time for Oblique Tree to train (s)")
plt.title("Oblique Tree Training time: Cython vs Python")
plt.legend()
plt.savefig("Performance")
print(cython_y)

