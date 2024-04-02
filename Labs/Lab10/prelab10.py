import numpy as np

def eval_legendre(n, x):
    if n == 0:
        p = [1]
    elif n == 1:
        p = [1, x]
    else:
        p_0 = 1
        p_1 = x
        p = [1, x]
        for i in range(2, n + 1):
            p_n = (1 / i) * ((2 * i - 1) * x * p_1 - (i - 1) * p_0 * x)
            p.append(p_n)
            p_0 = p_1
            p_1 = p_n

    return p


if __name__ == '__main__':
    print(eval_legendre(5 , 5))
