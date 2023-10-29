import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cholesky_factorization(A):
    n = A.shape[0]
    print(n)
    print(A)

A = np.array([[4, 1, 1, 1],
              [1, 3, -1, 1],
              [1, -1, 2, 0],
              [1, 1, 0, 2]])

cholesky_factorization(A)