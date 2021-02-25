import sys
import numpy as np
import pandas as pd
from proglearn import LifelongClassificationForest as lcf

def load_data(n):

  # They are labelled backwards!
  ftrain = "Sparse_parity_test.csv"
  ftest = "Sparse_parity_train.csv"

  df_train = pd.read_csv(ftrain, header=None).to_numpy()
  df_test = pd.read_csv(ftest, header=None).to_numpy()

  idx = np.random.choice(10000, n, replace=False)

  X_train = df_train[idx, :-1]
  y_train = df_train[idx, -1]

  X_test = df_test[:, :-1]
  y_test = df_test[:, -1]

  return X_train, y_train, X_test, y_test

def test(n, reps, n_estimators, feature_combinations, density):

  preds = np.zeros((reps, 1000))
  acc = np.zeros(reps)
  for i in range(reps):

    X_train, y_train, X_test, y_test = load_data(n)

    clf = lcf(default_n_estimators=n_estimators,
              oblique=True,
              default_feature_combinations=feature_combinations,
              default_density=density
              )

    clf.add_task(X_train, y_train)
    preds[i] = clf.predict(X_test, 0)
    acc[i] = np.sum(preds[i] == y_test) / len(y_test)

  np.save("output/sparse_parity_preds_" + str(n) + ".npy", preds)
  return acc


def main():

  n = 10000
  reps = 5
  n_estimators = 100
  feature_combinations = 1.5
  density = 0.25

  acc = test(n, reps, n_estimators, feature_combinations, density)
  print(acc)

if __name__ == "__main__":
  main()
