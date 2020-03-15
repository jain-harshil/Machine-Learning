import numpy as np
from preprocessing.polynomial_features_ import PolynomialFeatures

X = np.array([2,4])
poly = PolynomialFeatures(2,include_bias = True)
print(poly.transform(X))

