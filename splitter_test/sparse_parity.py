import sys
import numpy as np
import pandas as pd
import sporfdata as sd
#from proglearn import LifelongClassificationForest as lcf
#from sklearn.ensemble import RandomForestClassifier as rfc

from rerf.rerfClassifier import rerfClassifier as rfc

def load_data(n):

  ftrain = "Sparse_parity_test.csv"
  ftest = "Sparse_parity_test.csv"

  #df_train = pd.read_csv(ftrain, header=None).to_numpy()
  df_test = pd.read_csv(ftest, header=None).to_numpy()

  #idx = np.random.choice(10000, n, replace=False)

  #X_train = df_train[idx, :-1]
  #y_train = df_train[idx, -1]

  X_test = df_test[:, :-1]
  y_test = df_test[:, -1]

  X_train, y_train = sd.sparse_parity(n)

  return X_train, y_train, X_test, y_test

def test_rf(n, reps, n_estimators):

  preds = np.zeros((reps, 10000))
  acc = np.zeros(reps)
  for i in range(reps):

    X_train, y_train, X_test, y_test = load_data(n)

    clf = rfc(n_estimators=n_estimators)
            #projection_matrix="Base",
            #oob_score=False)
              #max_features=10)
    clf.fit(X_train, y_train)
    
    
    preds[i] = clf.predict(X_test)
    acc[i] = np.sum(preds[i] == y_test) / len(y_test)

  np.save("output/rf_sparse_parity_preds_" + str(n) + ".npy", preds)
  return acc

def test_lcf(n, reps, n_estimators, feature_combinations, density):

  preds = np.zeros((reps, 10000))
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

  #acc = test_lcf(n, reps, n_estimators, feature_combinations, density)
  acc = test_rf(n, reps, n_estimators)
  print(acc)

if __name__ == "__main__":
  main()
