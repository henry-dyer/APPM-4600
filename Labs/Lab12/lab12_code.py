import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy import linalg
from scipy.linalg import lu_factor, lu_solve
import time


def driver():
    ''' create  matrix for testing different ways of solving a square
     linear system'''
    '''' N = size of system'''
    N = 2000

    ''' Right hand side'''
    b = np.random.rand(N, 1)
    A = np.random.rand(N, N)

    start_time = time.perf_counter_ns()

    lu, piv = lu_factor(A)

    end_time = time.perf_counter_ns()
    elapsed_time = end_time - start_time
    print(f"LU factorization for N = {N} was {elapsed_time:.2f} nS.")

    start_time = time.perf_counter_ns()

    x = lu_solve((lu, piv), b)

    end_time = time.perf_counter_ns()
    elapsed_time = end_time - start_time
    print(f"LU solve time for N = {N} was {elapsed_time:.2f} nS.")

    test = np.matmul(A, x)
    r = la.norm(test - b)

    print('LU:', r, '\n')

    start_time = time.perf_counter_ns()
    x = scila.solve(A, b)
    end_time = time.perf_counter_ns()
    elapsed_time = end_time - start_time
    print(f"Standard Solve for N = {N} was {elapsed_time:.2f} nS.")

    test = np.matmul(A, x)
    r = la.norm(test - b)

    print('Normal:', r)

    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N, M)
    b = np.random.rand(N, 1)

    print('\n----------------\n')

    #standard

    x = la.inv(A.T @ A) @ A.T @ b

    print(x)

    Q, R = linalg.qr(A)

    x = la.inv(R) @ Q.T @ b

    print(x)




def create_rect(N, M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1, 10, M)
    d = 10 ** (-a)

    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N, N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1, R)
    A = np.random.rand(M, M)
    Q2, R = la.qr(A)
    test = np.matmul(Q2, R)

    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B




if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
