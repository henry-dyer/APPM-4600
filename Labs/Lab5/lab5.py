import numpy as np


def bisection(f, fp, fpp, a, b, Nmax):
    fa = f(a)
    fb = f(b)
    count = 0

    if fa * fb > 0:
        ier = 1
        astar = a
        return [astar, ier]

    if fa == 0:
        astar = a
        ier = 0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    while count < Nmax:
        c = 0.5 * (a + b)

        if (f(c) * fpp(c)) / (fp(c) ** 2) < 1:
            ier = 0
            return [c, ier]

        fc = f(c)

        if fc == 0:
            astar = c
            ier = 0
            return [astar, ier]
        if fa * fc < 0:
            b = c
        elif fb * fc < 0:
            a = c
            fa = fc
        else:
            astar = c
            ier = 3
            return [astar, ier]

        count += 1

    astar = a
    ier = 2
    return [astar, ier]


def newton(f, fp, p0, tol, Nmax):
    p = np.zeros(Nmax + 1)
    p[0] = p0

    it = 0

    for it in range(Nmax):
        p1 = p0 - f(p0) / fp(p0)
        p[it + 1] = p1
        if abs(p1 - p0) < tol:
            pstar = p1
            info = 0
            return [p, pstar, info, it]
        p0 = p1

    pstar = p1
    info = 1

    return [p, pstar, info, it]


def hybridNewton(f, fp, fpp, a, b, tol, Nmax):
    [astar, ier] = bisection(f, fp, fpp, a, b, Nmax)

    if ier == 0:
        return newton(f, fp, astar, tol, Nmax)
    else:
        print('Bisection did not locate midpoint in basin')
        return -1


f = lambda x: np.exp(x ** 2 + 7 * x - 30) - 1
fp = lambda x: (2 * x + 7) * np.exp(x ** 2 + 7 * x - 30)
fpp = lambda x: (4 * x ** 2 + 28 * x + 51) * np.exp(x ** 2 + 7 * x - 30)

a = 2
b = 4.5
tol = 10 ** -8
Nmax = 100

print(hybridNewton(f, fp, fpp, a, b, tol, Nmax))
