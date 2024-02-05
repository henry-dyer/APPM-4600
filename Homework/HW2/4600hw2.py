import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def question2B():
    A = 1 / 2 * np.array([[1, 1], [1 + 10 ** -10, 1 - 10 ** -10]])
    print(LA.cond(A))
    return


def question3B():
    x = 5**-10

    actual = np.expm1(x)
    estimate = np.exp(x) - 1

    for i in range(5, 11, 1):
        x = i ** -10
        actual = np.expm1(x)
        estimate = np.exp(x) - 1
        print('Relative Error:', np.abs(actual - estimate) / np.abs(actual))

    return


def question3C():
    S = 0
    n = 1
    x = 9.999999995000000 ** -10
    actual = np.expm1(x)

    while np.abs(actual - S) / np.abs(actual) >= 10 ** -16:
        S += x ** n / math.factorial(n)
        n += 1

    print(n)

    return


def question4A():
    t = np.linspace(0, np.pi, 31)
    y = np.cos(t)
    S = np.dot(t, y)

    print('the sum is:', S)

    return


def question4B(R=1.2, delta_r=0.1, f=15, p=float(0), theta=0):
    x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
    y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
    return [x, y]


def Driver4B():
    theta_values = np.linspace(0, 2 * np.pi, 1000)

    for i in range(1, 11):
        p = np.random.uniform(0, 2)

        values = np.array([question4B(i, 0.1, 2 + i, p, theta) for theta in theta_values])

        x_values = values[:, 0]
        y_values = values[:, 1]

        plt.plot(x_values, y_values, label=f'Curve {i}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.grid(True)
    plt.show()

    return


question3C()




