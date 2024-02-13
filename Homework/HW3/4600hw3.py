import numpy as np
import matplotlib.pyplot as plt


# Q5A
def f5A(x):
    return x - 4 * np.sin(2 * x) - 3


x = np.linspace(-8, 8, 400)

y = f5A(x)
plt.axhline(0, color='black', linewidth=0.5)
plt.plot(x, y)
plt.title('5A')
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.show()

# Q5B
def fixedpt(f, x0, tol, Nmax):
    count = 0
    while count < Nmax or (f()):
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


f = lambda x: -np.sin(2 * x) + 1.25 * x - .75

Nmax = 100
tol = 0.5 * 10**-10
''' test f '''
x0 = 5
[xstar, ier] = fixedpt(f, x0, tol, Nmax)
print('the approximate fixed point is:', xstar)
print('f1(xstar):', f(xstar))
print('Error message reads:', ier)


