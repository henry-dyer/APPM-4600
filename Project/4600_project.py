import numpy as np
import matplotlib.pyplot as plt

''' 
f : function we are analyzing
a : left bound for data points
b : right bound for data points
N : number of data points to generate
noise: whether errors for data points are generated from a 'normal' or 'uniform' distribution
sigma: standard deviation for error generation
points: whether x coordinates from points are equispaced across [a,b] or randomly from a uniform distribution on [a,b]
'''
def generate_data(f, a, b, N, noise='normal', sigma=1, points='equispaced'):
    if points == 'equispaced':
        x = np.linspace(a, b, N)
    elif points == 'random':
        x = np.random.uniform(a, b, N)
    else:
        print('Invalid Point Input')
        exit(0)

    if noise == 'normal':
        residuals = np.random.normal(0, sigma, N)
    elif noise == 'uniform':
        bounds = sigma * np.sqrt(3)
        residuals = np.random.uniform(-bounds, bounds, N)
    else:
        print('Invalid Error Input')
        exit(0)

    y = f(x) + residuals

    return np.array(list(zip(x, y)))


def plot(func, reg, data, a, b):
    x = np.linspace(a, b, 100)

    plt.plot(x, func(x), label='Underlying Function', color='green', linestyle='--')
    plt.plot(x, reg(x), label='Regression Line', color='blue')
    plt.scatter(data[:, 0], data[:, 1], color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Underlying Function vs Regression Line')
    plt.legend()
    plt.show()

    plt.plot(x, func(x) - reg(x), label='Error in Underlying Function', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Error in Regression Line')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':

    f = lambda x: x * np.sin(x) ** 2
    reg = lambda x: x**2 + 2*x + 3

    a = 0
    b = 10
    N = 10

    data = generate_data(f, a, b, N, sigma=3, points='equispaced')

    plot(f, reg, data, a, b)





