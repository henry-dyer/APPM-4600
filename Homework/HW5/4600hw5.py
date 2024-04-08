import numpy as np
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def question1(x_values, y_values, z):
    n = len(x_values)
    weights = np.zeros(n)

    for j in range(n):
        products = np.ones(n)
        for i in range(n):
            if i != j:
                products[i] = (x_values[j] - x_values[i])
        weights[j] = 1 / np.prod(products)

    numerator = 0
    denominator = 0

    for i in range(n):
        numerator += y_values[i] * weights[i] / (z - x_values[i])
        denominator += weights[i] / (z - x_values[i])

    return numerator / denominator


def q1b_driver():
    f = lambda x: 1 / (1 + (16 * x) ** 2)

    n = 17

    x_vals = np.zeros(n + 1)

    for i in range(1, n + 2):
        x_vals[i - 1] = - 1 + ((i - 1) * (2 / n))

    y_vals = f(x_vals)

    z_vals = np.linspace(-1, 1, 1001)

    y_interp = np.zeros(len(z_vals))

    for i in range(1001):
        y_interp[i] = question1(x_vals, y_vals, z_vals[i])

    plt.plot(x_vals, y_vals, 'o',label='Interpolation Points', color='red')
    plt.plot(z_vals, y_interp, label=f'Barycentric Lagrange with n = {n}', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Actual vs Barycentric')
    plt.legend()
    plt.show()

    return


def q1c_driver():
    f = lambda x: 1 / (1 + (16 * x) ** 2)

    n = 17

    x_vals = np.zeros(n)

    for i in range(0, n):
        x_vals[i] = np.cos(((2 * i + 1) * np.pi) / (2 * n + 2))

    y_vals = f(x_vals)

    z_vals = np.linspace(-1, 1, 1001)

    y_interp = np.zeros(len(z_vals))

    for i in range(1001):
        y_interp[i] = question1(x_vals, y_vals, z_vals[i])

    plt.plot(x_vals, y_vals, 'o', label='Interpolation Points', color='red')
    plt.plot(z_vals, y_interp, label=f'Barycentric Lagrange for n = {n}', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Barycentric Interpolation with Chebyshev Points')
    plt.legend()
    plt.show()

    return

def question2A():

    A = np.array([[1, -1, 1], [1, 1, 1], [2, 1, 0]])

    b_0 = np.array([1, 0, 0])
    b_1 = np.array([0, 1, 0])
    b_2 = np.array([0, 0, 1])

    x_0 = solve(A, b_0)
    x_1 = solve(A, b_1)
    x_2 = solve(A, b_2)

    print('x_0:', x_0)
    print('x_1:', x_1)
    print('x_2:', x_2)

    return


def q3_spline(n):
    x = np.linspace(0, 1, n)
    y = np.sin(9 * x)

    x = np.concatenate((x, [2 * np.pi]))
    y = np.concatenate((y, [y[0]]))

    spline = CubicSpline(x, y, bc_type='periodic')

    return spline


def question3C():

    spline_05 = q3_spline(5)
    spline_10 = q3_spline(10)
    spline_20 = q3_spline(20)
    spline_40 = q3_spline(40)


    # Plot the original function
    x_values = np.linspace(0, 1, 100)
    '''plt.plot(x_values, np.sin(9 * x_values), label='sin(9x)', color='blue')

    # Plot the periodic cubic spline interpolation
    plt.plot(x_values, spline(x_values), label='Periodic Cubic Spline', color='red')

    

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Periodic Cubic Spline Interpolation of sin(9x)')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()'''

    original_values = np.sin(9 * x_values)
    interp_values_05 = spline_05(x_values)
    interp_values_10 = spline_10(x_values)
    interp_values_20 = spline_20(x_values)
    interp_values_40 = spline_40(x_values)

    # Compute the absolute error between the original and interpolation functions
    absolute_error_05 = np.abs(original_values - interp_values_05)
    absolute_error_10 = np.abs(original_values - interp_values_10)
    absolute_error_20 = np.abs(original_values - interp_values_20)
    absolute_error_40 = np.abs(original_values - interp_values_40)

    # Plot the log of the error
    plt.semilogy(x_values, absolute_error_05, label='n=5', color='blue')
    plt.semilogy(x_values, absolute_error_10, label='n=10', color='red')
    plt.semilogy(x_values, absolute_error_20, label='n=20', color='green')
    plt.semilogy(x_values, absolute_error_40, label='n=40', color='orange')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Plot of Abs Error Between Periodic Cubic Spline and True Value for sin(9x)')
    plt.legend()
    plt.show()

    return


def question4A():
    A = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    b = np.array([1, 4, 2, 6])

    x = inv(A.transpose() @ A) @ A.transpose() @ b

    return x


def question4B():
    A = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    b = np.array([1, 4, 2, 6])
    D = np.diag([1, 2, 3, np.sqrt(6)])

    x = inv(A.transpose() @ D @ A) @ A.transpose() @ D @ b

    return x


def Q4_driver():
    ols = question4A()

    wls = question4B()

    data = np.array([[0, 1], [1, 4], [2, 2], [3, 6]])
    X = data[:, 0]  # Features (x)
    y = data[:, 1]

    plt.scatter(X, y, label='Data')

    # Plot the regression line
    x_values = np.linspace(min(X), max(X), 100)
    y_ols = ols[0] * x_values + ols[1]
    y_wls = wls[0] * x_values + wls[1]
    plt.plot(x_values, y_ols, color='red', label='OLS Regression Line')
    plt.plot(x_values, y_wls, color='blue', label='WLS Regression Line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('OLS vs WLS')
    plt.legend()

    plt.grid(True)
    plt.show()

    return


if __name__ == '__main__':

    #q1b_driver()

    q1c_driver()

    #question2A()

    #question3C()

    #Q4_driver()