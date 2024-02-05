import numpy as np
import numpy.linalg as la
import math

def driver():
    n = 100
    x = np.linspace(0, np.pi, n)

    f = lambda x: np.cos(x)
    g = lambda x: np.cos(2 * x)
    y = f(x)
    w = g(x)

    # evaluate the dot product of y and w
    dp = dotProduct(y, w, n)
    # print the output
    print('The dot product is:', dp)
    return


#   Computes the dot product of the n x 1 vectors x and y

def dotProduct(x, y, n):
    dp = 0
    for j in range(n):
        dp = dp + x[j] * y[j]
    return dp


A = np.array([[1, 2, 3, 4], [4, 5, 4, 10]])
x = np.array([1, 2, 10, 35])

def matrixVectorProduct(A, x):
    if A.shape[1] != x.shape[0]:
        print('Incompatible System')
        return -1

    b = np.zeros([A.shape[0], 1])
    for i in range(A.shape[0]):
        b[i] = dotProduct(A[i], x, A.shape[1])
    return b


print(np.matmul(A, x))


print(matrixVectorProduct(A, x))


# driver()
