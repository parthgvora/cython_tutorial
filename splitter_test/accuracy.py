import numpy as np
from proglearn import LifelongClassificationForest as lcf
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
clf = lcf(oblique=True)
clf.add_task(X_iris, y_iris)
y_hat = clf.predict(X_iris, 0)
acc = np.sum(y_hat == y_iris)
print("iris: ", acc / len(y_iris))

breast_cancer = load_breast_cancer()
X_breast_cancer, y_breast_cancer = breast_cancer.data, breast_cancer.target
clf = lcf(oblique=True)
clf.add_task(X_breast_cancer, y_breast_cancer)
y_hat = clf.predict(X_breast_cancer, 0)
acc = np.sum(y_hat == y_breast_cancer)
print("breast_cancer: ", acc / len(y_breast_cancer))


wine = load_wine()
X_wine, y_wine = wine.data, wine.target
clf = lcf(oblique=True)
clf.add_task(X_wine, y_wine)
y_hat = clf.predict(X_wine, 0)
acc = np.sum(y_hat == y_wine)
print("wine: ", acc / len(y_wine))

diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
clf = lcf(oblique=True)
clf.add_task(X_diabetes, y_diabetes)
y_hat = clf.predict(X_diabetes, 0)
acc = np.sum(y_hat == y_diabetes)
print("diabetes: ", acc / len(y_diabetes))
