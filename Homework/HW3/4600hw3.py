import numpy as np
import matplotlib.pyplot as plt


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
            print('Number of iterations:', count)
            astar = a
            ier = 0
            return [astar, ier]
        count += 1

    astar = a
    ier = 2
    return [astar, ier]


# Q3B

f3b = lambda x: x ** 3 + x - 4

a = 1
b = 4
Nmax = 100
tol = 10 ** -3
[astar, ier] = bisection(f3b, a, b, tol, Nmax)
print('the approximate root is', astar)
print('the error message reads:', ier)


# Q5A
def f5A(x):
    return x - 4 * np.sin(2 * x) - 3


x = np.linspace(-8, 8, 400)

y = f5A(x)
plt.axhline(0, color='black', linewidth=0.5)
plt.plot(x, y)
plt.title('Question 5A Plot')
plt.xlabel('x')
plt.ylabel('f(x) = x- 4sin(2x) - 3')
#plt.show()

# Q5B

f5b = lambda x: -np.sin(2 * x) + (1.25 * x) - .75

Nmax = 100
tol = 0.5 * 10 ** -10
x0 = 0
[xstar, ier] = fixedpt(f5b, x0, tol, Nmax)
print('the approximate fixed point is:', xstar)
print('f5b(xstar):', f5b(xstar))
print('Error message reads:', ier)
