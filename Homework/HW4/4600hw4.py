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


def q3c_fun(x):
    f_x = np.zeros(3)
    f_x[0] = x[0] + np.cos(x[0] * x[1] * x[2]) - 1
    f_x[1] = (1 - x[0]) ** .25 + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1
    f_x[2] = -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1
    return f_x[0]**2 + f_x[1]**2 + f_x[2]**2

    # gradient vector
def q3c_Gfun(x):
    J = np.zeros((3, 3))

    J[0, 0] = 2 * (x[0] + np.cos(x[0] * x[1] * x[2]) - 1) * (1 - x[1] * x[2] * np.sin(x[0] * x[1] * x[2]))
    J[0, 1] = 2 * (x[0] + np.cos(x[0] * x[1] * x[2]) - 1) *(- x[0] * x[2] * np.sin(x[0] * x[1] * x[2]))
    J[0, 2] = 2 * (x[0] + np.cos(x[0] * x[1] * x[2]) - 1) *(- x[0] * x[1] * np.sin(x[0] * x[1] * x[2]))
    J[1, 0] = 2 * ((1 - x[0]) ** .25 + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1) * (-0.25 * (1 - x[0]) ** -0.75)
    J[1, 1] = 2 * ((1 - x[0]) ** .25 + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1)
    J[1, 2] = 2 * ((1 - x[0]) ** .25 + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1) * (0.1 * x[2] - 0.15)
    J[2, 0] = 2 * (-x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1) * (-2 * x[0])
    J[2, 1] = 2 * (-x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1) *(-0.2 * x[1] + 0.01)
    J[2, 2] = 2 * (-x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1)

    return np.array([J[0, 0] + J[0, 1] + J[0, 2], J[1, 0] + J[1, 1] + J[1, 2], J[2, 0] + J[2, 1] + J[2, 2]])


def line_search(f, Gf, x0, p, type, mxbck, c1, c2):
    alpha = 2
    n = 0
    cond = False  # condition (if True, we accept alpha)
    f0 = f(x0)  # initial function value
    Gdotp = p.T @ Gf(x0)  # initial directional derivative
    nf = 1
    ng = 1  # number of function and grad evaluations

    # we backtrack until our conditions are met or we've halved alpha too much
    while n <= mxbck and (not cond):
        alpha = 0.5 * alpha
        x1 = x0 + alpha * p

        Armijo = f(x1) <= f0 + c1 * alpha * Gdotp
        nf += 1
        if type == 'wolfe':

            Curvature = p.T @ Gf(x1) >= c2 * Gdotp
            # condition is sufficient descent AND slope reduction
            cond = Armijo and Curvature
            ng += 1
        elif type == 'swolfe':
            Curvature = np.abs(p.T @ Gf(x1)) <= c2 * np.abs(Gdotp)
            # condition is sufficient descent AND symmetric slope reduction
            cond = Armijo and Curvature
            ng += 1
        else:
            # Default is Armijo only (sufficient descent)
            cond = Armijo

        n += 1

    return x1, alpha, nf, ng


################################################################################
# Steepest descent algorithm
def steepest_descent(f, Gf, x0, tol, nmax, type='swolfe', verb=True):
    # Set linesearch parameters
    c1 = 1e-3
    c2 = 0.9
    mxbck = 10

    # Initialize alpha, fn and pn
    alpha = 1
    xn = x0  # current iterate
    rn = x0  # list of iterates
    fn = f(xn)
    nf = 1  # function eval
    pn = -Gf(xn)
    ng = 1  # gradient eval

    # if verb is true, prints table of results

    # while the size of the step is > tol and n less than nmax
    n = 0
    while n <= nmax and np.linalg.norm(pn) > tol:

        # Use line_search to determine a good alpha, and new step xn = xn + alpha*pn
        (xn, alpha, nfl, ngl) = line_search(f, Gf, xn, pn, type, mxbck, c1, c2)

        nf = nf + nfl
        ng = ng + ngl  # update function and gradient eval counts
        fn = f(xn)  # update function evaluation
        pn = -Gf(xn)  # update gradient evaluation
        n += 1
        rn = np.vstack((rn, xn))  # add xn to list of iterates

    r = xn  # approx root is last iterate

    return r, rn, nf, ng


def q3_hybrid():
    x0 = np.array([0, 0, 0])
    tol = 1e-2
    Nmax = 100

    (r, rn, nf, ng) = steepest_descent(q3c_fun, q3c_Gfun, x0, tol, Nmax)

    tol = 1e-6

    return Q3_Newton(r, tol, Nmax)


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

    print('\n------------------------------------------\n')

    t = time.time()
    for j in range(20):
        (r, rn, nf, ng) = steepest_descent(q3c_fun, q3c_Gfun, x0, tol, Nmax)
    elapsed = time.time() - t
    print(r)
    print('Q3 Steepest Decent: took this many seconds:', elapsed / 20)

    print('\n------------------------------------------\n')

    t = time.time()
    for j in range(20):
        [xstar, ier, its] = q3_hybrid()
    elapsed = time.time() - t
    print(xstar)
    print('Q3 Hybrid: took this many seconds:', elapsed / 20)

