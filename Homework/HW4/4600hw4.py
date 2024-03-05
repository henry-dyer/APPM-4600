import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm


# Code for Question 1
def eval_Q1F(x):
    return np.array([3 * x[0] ** 2 - x[1] ** 2, 3 * x[0] * x[1] ** 2 - x[0] ** 3 - 1])


def eval_Q1J(x):
    J = np.zeros((2, 2))

    J[0, 0] = 6 * x[0]
    J[0, 1] = -2 * x[1]
    J[1, 0] = 3 * x[1] ** 2 - 3 * x[0] ** 2
    J[1, 1] = 6 * x[0] * x[1]

    return J


def question1A(x0, tol, Nmax):
    for its in range(Nmax):

        j_inv = np.array([[1 / 6, 1 / 18], [0, 1 / 6]])

        F = eval_Q1F(x0)
        x1 = x0 - np.matmul(j_inv, F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Q1_Newton(x0, tol, Nmax):
    for its in range(Nmax):
        J = eval_Q1J(x0)

        Jinv = inv(J)
        F = eval_Q1F(x0)

        x1 = x0 - Jinv.dot(F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


# ----------------------------------
# Code for Question 2
def eval_Q2F(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 4, np.exp(x[0]) + x[1] - 1])


def eval_Q2J(x):
    J = np.zeros((2, 2))

    J[0, 0] = 2 * x[0]
    J[0, 1] = 2 * x[1]
    J[1, 0] = np.exp(x[0])
    J[1, 1] = 1

    return J


def Q2_Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = eval_Q2J(x0)
        Jinv = inv(J)
        F = eval_Q2F(x0)

        x1 = x0 - Jinv.dot(F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Q2_LazyNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = eval_Q2J(x0)
    Jinv = inv(J)
    for its in range(Nmax):

        F = eval_Q2F(x0)
        x1 = x0 - Jinv.dot(F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Q2_Broyden(x0, tol, Nmax):
    A0 = eval_Q2J(x0)

    v = eval_Q2F(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0 + s
    for its in range(Nmax):
        w = v
        ''' create new v'''
        v = eval_Q2F(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v - w;
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s, z)
        u = np.dot(s, A)
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s + z
        tmp2 = np.outer(tmp, u)
        A = A + 1. / p * tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk + s
        if norm(s) < tol:
            alpha = xk
            ier = 0
            return [alpha, ier, its]
    alpha = xk
    ier = 1
    return [alpha, ier, its]


# ----------------------------------
# Code for Question 3

def eval_Q3F(x):
    f = x[0] + np.cos(x[0] * x[1] * x[2]) - 1
    g = (1 - x[0]) ** .25 + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1
    h = -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1
    return np.array([f, g, h])


def eval_Q3J(x):
    J = np.zeros((3, 3))

    J[0, 0] = 1 + x[1] * x[2] * np.cos(x[0] * x[1] * x[2])
    J[0, 1] = x[0] * x[2] * np.cos(x[0] * x[1] * x[2])
    J[0, 2] = x[0] * x[1] * np.cos(x[0] * x[1] * x[2])
    J[1, 0] = -0.25 * (1 - x[0]) ** -0.75
    J[1, 1] = 1
    J[1, 2] = 0.1 * x[2] - 0.15
    J[2, 0] = -2 * x[0]
    J[2, 1] = -0.2 * x[1] + 0.01
    J[2, 2] = 1

    return J


def Q3_Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = eval_Q3J(x0)
        Jinv = inv(J)
        F = eval_Q3F(x0)

        x1 = x0 - Jinv.dot(F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]



if __name__ == '__main__':
    x0 = np.array([1, 1])

    Nmax = 100
    tol = 1e-10

    print('Question 1A Results \n')

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = question1A(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Question 1A: the error message reads:', ier)
    print('Question 1A: took this many seconds:', elapsed / 20)
    print('Question 1A: number of iterations is:', its)

    print('\n------------------------------------------\n')
    print('Question 1C Results \n')

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = Q1_Newton(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Newton: the error message reads:', ier)
    print('Newton: took this many seconds:', elapsed / 20)
    print('Netwon: number of iterations is:', its)

    print('\n------------------------------------------\n')
    print('Question 2i Results \n')

    x0 = np.array([1, 1])

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = Q2_Newton(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Q2 Newton: the error message reads:', ier)
    print('Q2 Newton: took this many seconds:', elapsed / 20)
    print('Q2 Netwon: number of iterations is:', its)

    print('\n------------------------------------------\n')

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = Q2_LazyNewton(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Q2 Lazy Newton: the error message reads:', ier)
    print('Q2 Lazy Newton: took this many seconds:', elapsed / 20)
    print('Q2 Lazy Newton: number of iterations is:', its)

    print('\n------------------------------------------\n')

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = Q2_Broyden(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Q2 Broyden: the error message reads:', ier)
    print('Q2 Broyden: took this many seconds:', elapsed / 20)
    print('Q2 Broyden: number of iterations is:', its)

    print('\n------------------------------------------\n')
    print('Question 3 Answers')

    x0 = np.array([0, 0, 0])
    tol = 1e-6

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = Q3_Newton(x0, tol, Nmax)
    elapsed = time.time() - t
    print(xstar)
    print('Q3 Newton: the error message reads:', ier)
    print('Q3 Newton: took this many seconds:', elapsed / 20)
    print('Q3 Netwon: number of iterations is:', its)
