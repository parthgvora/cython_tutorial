import sys
import numpy as np
from sklearn.datasets import load_iris

from rerf.rerfClassifier import rerfClassifier as rfc
from proglearn.transformers import ObliqueTreeClassifier as of

data = load_iris()
X, y = data.data, data.target

rerf_acc = np.zeros(5)
of_acc = np.zeros(5)
for i in range(5):
    OF = of(#oblique=True, 
            max_features=1.0, 
            feature_combinations=1.0)

    RFC = rfc(n_estimators=1, 
        projection_matrix="RerF", 
        feature_combinations=1.0, 
        max_features=1.0)

    # Rerf
    RFC.fit(X, y)
    rerf_preds = RFC.predict(X)
    rerf_acc[i] = np.sum(rerf_preds == y) / len(y)

    # Of
    OF.fit(X, y)
    of_preds = OF.predict(X)
    of_acc[i] = np.sum(of_preds == y) / len(y)
    
    
print("RerF: ", np.mean(rerf_acc))
print("OF: ", np.mean(of_acc))


