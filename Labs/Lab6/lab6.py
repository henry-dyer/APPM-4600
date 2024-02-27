import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import time


def centered_difference(f, x0, y0, x1, y1, h):
    return (f(x0, y0) - f(x1, y1)) / (2 * h)


def evalF(x):
    F = np.zeros(3)
    F[0] = 3*x[0]-np.cos(x[1]*x[2])-1/2
    F[1] = x[0]-81*(x[1]+0.1)**2+np.sin(x[2])+1.06
    F[2] = np.exp(-x[0]*x[1])+20*x[2]+(10*np.pi-3)/3
    return F


def evalJ(x):

    J = np.array([[3.0, x[2]*np.sin(x[1]*x[2]), x[1]*np.sin(x[1]*x[2])],
        [2.*x[0], -162.*(x[1]+0.1), np.cos(x[2])],
        [-x[1]*np.exp(-x[0]*x[1]), -x[0]*np.exp(-x[0]*x[1]), 20]])
    return J


def evalF1(x):
    F = np.zeros(2)
    F[0] = 4 * x[0] ** 2 + x[1] ** 2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] -x[1])
    return F


def evalJ1(x):
    J = np.zeros((2, 2))
    J[0, 0] = 8 * x[0]
    J[0, 1] = 8 * x[1]
    J[1, 0] = 1 - np.cos(x[0] - x[1])
    J[1, 1] = 1 + np.cos(x[0] - x[1])
    return J


def evalJ_finite(x, h):
    J = np.zeros((2, 2))

    J[0, 0] = evalF1([x[0] + h, x[1]])[0] - evalF1([x[0] - h, x[1]])[0] / (2 * h)
    J[0, 1] = evalF1([x[0], x[1] + h])[0] - evalF1([x[0], x[1] - h])[0] / (2 * h)
    J[1, 0] = evalF1([x[0] + h, x[1]])[1] - evalF1([x[0] - h, x[1]])[1] / (2 * h)
    J[1, 1] = evalF1([x[0], x[1] + h])[1] - evalF1([x[0], x[1] - h])[1] / (2 * h)

    return J



def LazyNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''



    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        if its % 3 == 0:
            J = evalJ(x0)
            Jinv = inv(J)

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Newton(x0, tol, Nmax, h):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJ_finite(x0, h)
        Jinv = inv(J)
        F = evalF1(x0)

        x1 = x0 - Jinv.dot(F)

        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


x0 = np.array([1, 0])

h = 1 * 10 ** -7

Nmax = 100
tol = 1e-10

"""t = time.time()
for j in range(20):
  [xstar,ier,its] = LazyNewton(x0,tol,Nmax)
elapsed = time.time()-t
print(xstar)
print('Lazy Newton: the error message reads:',ier)
print('Lazy Newton: took this many seconds:',elapsed/20)
print('Lazy Newton: number of iterations is:',its)"""

t = time.time()
for j in range(20):
  [xstar,ier,its] = Newton(x0,tol,Nmax, h)
elapsed = time.time()-t
print(xstar)
print('Newton: the error message reads:',ier)
print('Newton: took this many seconds:',elapsed/20)
print('Netwon: number of iterations is:',its)
