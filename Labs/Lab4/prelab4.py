import numpy as np


def fixedpt(f, x0, tol, Nmax):
    count = 0
    x = np.empty([0, 0])
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            x = np.append(x, xstar)
            return x
        x0 = x1
    xstar = x1
    x = np.append(x, xstar)
    ier = 1
    return x


# use routines
f1 = lambda x: .5 * x
Nmax = 100
tol = 1e-6
''' test f1 '''

x0 = 1
x = fixedpt(f1, x0, tol, Nmax)
print('the approximate fixed point is:', x)
print('f1(xstar):', f1(x))