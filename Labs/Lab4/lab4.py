import numpy as np


def fixedpt(f, x0, tol, Nmax):
    count = 0
    x1 = np.NAN
    x = np.zeros((Nmax, 1))
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        x[count] = x1
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return x[: count + 1]
        x0 = x1
    xstar = x1
    x[count] = x1
    ier = 1
    return x


def aitkens(x, tol):
    p = np.empty((len(x) - 2, 1))
    if len(x) < 3:
        print('To Few Iterations')
        return -1

    for i in range(0, len(x) - 2):
        p[i] = (x[i] * x[i + 2] - x[i + 1] ** 2) / (x[i] - 2 * x[i + 1] + x[i + 2])
        if abs(p[i] - p[i - 1]) < tol:
            return p[:i + 1]

    return p


def steffensons(g, p0, tol, Nmax):
    count = 0
    p1 = np.NAN
    while count < Nmax:
        count += 1
        a = p0
        b = g(a)
        c = g(b)
        p1 = a - ((b - a) ** 2) / (c - 2 * b + a)
        if abs(p1 - p0) < tol:
            print('Steffensons number of iterations:', count)
            return p1
        else:
            p0 = p1

    return p1


f1 = lambda x1: (10 / (x1 + 4)) ** (1/2)
Nmax = 100
tol = 1e-10

x0 = 1.5
x = fixedpt(f1, x0, tol, Nmax)
aitkens_results = aitkens(x, tol)

print(x)

print('___')

print(aitkens_results)

print(steffensons(f1, x0, tol, Nmax))
