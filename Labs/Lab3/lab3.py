import numpy as np

def bisection(f, a, b, tol, Nmax):
    ''' first verify there is a root we can find in the interval '''
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        ier = 1
        astar = a
        return [astar, ier]

    ''' verify end point is not a root '''
    if fa == 0:
        astar = a
        ier = 0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    while count < Nmax:
        c = 0.5 * (a + b)
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
        if abs(b - a) < tol:
            astar = a
            ier = 0
            return [astar, ier]
        count += 1

    astar = a
    ier = 2
    return [astar, ier]


def fixedpt(f, x0, tol, Nmax):
    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]


#1A
f1 = lambda x: x * ((1 + (7 - x ** 5) / (x ** 2)) ** 3)

a1 = 1
b1 = 2
Nmax = 100
tol = 1e-5

print(fixedpt(f1, a1, tol, Nmax))







