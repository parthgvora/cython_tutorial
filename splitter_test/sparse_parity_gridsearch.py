import sys
import numpy as np
import pandas as pd

from rerf.rerfClassifier import rerfClassifier as rfc

# Sparse parity
reps = 3
n_estimators = 100
feature_combinations = [2, 3, 4, 5]
max_features = [3, 5, 10, 20] # ceil of 20 ^ (0.25, 0.5, 0.75, 1)

dftrain = np.load("data/sparse_parity_train_1000.npy")
dftest = np.load("data/sparse_parity_test.npy")

X_train = dftrain[:, :-1]
y_train = dftrain[:, -1]

X_test = dftest[:, :-1]
y_test = dftest[:, -1]

param_acc = np.zeros((4, 4))

for i, f in enumerate(feature_combinations):
    for j, m in enumerate(max_features):
    
        for k in range(3):

            clf = rfc(n_estimators=n_estimators,
                      projection_matrix="RerF",
                      feature_combinations=f,
                      max_features=m
                      )

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            param_acc[i, j] = np.sum(preds == y_test) / len(y_test)

print(param_acc)
np.save("sparse_parity_gridsearch", param_acc)

