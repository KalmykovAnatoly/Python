import numpy as np


def matrix_solve(A):
    n = A.shape[0]
    m = A.shape[1] - 1
    n -= 1

    try:
        for i in range(n):
            for j in range(n - i):
                A[n - j] = A[n - j] - A[i] * (A[n - j][i] / A[i][i])
        for i in range(n):
            for j in range(n - i):
                A[i] = A[i] - A[i + j + 1] * (A[i][i + j + 1] / A[i + j + 1][i + j + 1])

        answer = []
        for i in range(n + 1):
            if not np.any(A[i, :m]):
                raise RuntimeError
            if i <= m:
                answer.append(float(A[i][m] / A[i][i]))

        print("YES")
        for val in answer:
            print(val, end=" ")
    except:
        print("NO")
