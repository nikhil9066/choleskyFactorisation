import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cholesky_factorization(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_term = sum(L[i][k] ** 2 for k in range(i))
                L[i][i] = np.sqrt(A[i][i] - sum_term)
            else:
                sum_term = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (1 / L[j][j]) * (A[i][j] - sum_term)
    # print(L)
    U = np.transpose(L)
    # print(U)
    return L, U

def forward_substitution(L, b):
    n = L.shape[0]
    c = np.zeros(n)
    for i in range(n):
        c[i] = (b[i] - np.dot(L[i, :i], c[:i])) / L[i, i]
    # print(c)
    return c

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    # print(x)
    return x

A = np.array([[4, 1, 1, 1],
              [1, 3, -1, 1],
              [1, -1, 2, 0],
              [1, 1, 0, 2]])

b = np.array([0.65, 0.05, 0, 0.5])

# Cholesky factorization
L, U = cholesky_factorization(A)

# Forward substitution to solve Lc = b
y = forward_substitution(L, b)

# Backward substitution to solve Ux = y
x = backward_substitution(U, y)

print("Solution to the linear system:")
print(x)
