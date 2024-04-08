import numpy as np

def trapazoidal_quadrature(a, b, f, N):
    h = (b - a) / N
    integral = 0

    for i in range(1, N + 1):
        integral += (f(a + i * h) + f(a + (i - 1) * h)) * h

    return integral / 2


if __name__ == '__main__':
    f = lambda x: x ** 2
    a = 0
    b = 5
    N = 10

    print(trapazoidal_quadrature(a, b, f, N))