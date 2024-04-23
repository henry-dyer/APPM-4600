import numpy as np
from scipy.special import gamma
from scipy import integrate


def trapazoidal_quad(f, a, b, M):
    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)
    w = h * np.ones(M)
    w[0] = 0.5 * w[0]
    w[M - 1] = 0.5 * w[M - 1]

    return np.sum(f(x) * w)


def simpson_quad(f, a, b, M):
    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)
    w = (h / 3) * np.ones(M)
    w[1:M:2] = 4 * w[1:M:2]
    w[2:M - 1:2] = 2 * w[2:M - 1:2]

    return np.sum(f(x) * w)


def q2a():
    f = lambda x: 1 / (1 + x**2)
    a = -5
    b = 5
    M = 20

    print(trapazoidal_quad(f, a, b, M))
    print(simpson_quad(f, a, b, M))

    return


def trapazoidal_gamma(t, a, b, M):
    gamma = lambda x: x ** (t - 1) * np.exp(-x)

    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)
    w = h * np.ones(M)
    w[0] = 0.5 * w[0]
    w[M - 1] = 0.5 * w[M - 1]

    return np.sum(gamma(x) * w)


def q4a():
    t = 2
    a = 0
    b = t * 5
    M = t * 25


    print(trapazoidal_gamma(t, a, b, M))
    print(gamma(t))


def q4b():
    t = 2
    a = 0
    b = t * 10

    gamma = lambda x: x ** (t - 1) * np.exp(-x)

    I, err, infodict = integrate.quad(gamma, a, b)

    return


def q4c():
    t = 4
    x, w = np.polynomial.laguerre.laggauss(t)

    f = lambda x: x ** (t - 1)

    I = 0
    for i in range(t):
        I += w[i] * f(x[i])

    print(I)
    return


if __name__ == '__main__':
    q4c()

