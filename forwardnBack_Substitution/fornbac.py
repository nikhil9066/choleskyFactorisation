import numpy as np
import warnings
warnings.filterwarnings("ignore")

def forward_substitution(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

L = np.array([[1, 0, 0],
              [2, 3, 0],
              [4, 5, 6]])

b = np.array([3, 8, 18])
print(L,"\n")
print(b,"\n")
x1 = forward_substitution(L, b)
print("Solution to Lx = b:", x1)
print("-----------------------------\n")