import numpy as np
import warnings
warnings.filterwarnings("ignore")

def forward_substitution(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

def backward_substitution(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
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

U = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]])

b = np.array([3, 8, 18])
print(U,"\n")
print(b,"\n")
x2 = backward_substitution(U, b)
print("Solution to Ux = b:", x2)