import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cholesky_factorization(A):
    n = A.shape[0]
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            if i == j:
                sum_term = sum(R[k, j] ** 2 for k in range(j))
                R[j, j] = np.sqrt(A[j, j] - sum_term)
            else:
                sum_term = sum(R[k, j] * R[k, i] for k in range(j))
                R[j, i] = (1 / R[j, j]) * (A[j, i] - sum_term)

    return R

A_int = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
R_int = cholesky_factorization(A_int)
print("Cholesky Factor Integer variable:")
print(R_int)

print("----------Another float variable example-------------\n")

A_float = np.array([[9, 3, 6],
                    [3, 25, -7],
                    [6, -7, 49]], dtype=float)
R_float = cholesky_factorization(A_float)
print("Cholesky Factor float variable:")
print(R_float)

print("----------Another complex variable example-------------\n")

A_complex = np.array([[9 + 4j, 3 - 1j, 6],
                     [3 - 1j, 25, -7],
                     [6, -7, 49 + 2j]], dtype=complex)
R_complex = cholesky_factorization(A_complex)
print("Cholesky Factor for complex variable:")
print(R_complex)