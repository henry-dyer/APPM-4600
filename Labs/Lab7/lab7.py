import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


def driver():
    #f = lambda x: (1 + (10 * x)**2) ** -1
    f = lambda x: np.sinc(5 * x)

    N = 20
    ''' interval'''
    a = -1
    b = 1

    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a, b, N + 1)

    for i in range(N + 1):
        xint[i] = np.cos(((2 * i) * np.pi) / (2 * N + 2))


    ''' create interpolation data'''
    yint = f(xint)

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval_l = np.zeros(Neval + 1)
    yeval_dd = np.zeros(Neval + 1)
    yeval_mm = np.zeros(Neval + 1)

    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        y[j][0] = yint[j]

    y = dividedDiffTable(xint, y, N + 1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk], xint, yint, N)
        yeval_dd[kk] = evalDDpoly(xeval[kk], xint, y, N)
        yeval_mm[kk] = eval_monomial(xint, yint, xeval[kk])

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval, fex, 'ro-')
    plt.plot(xeval, yeval_l, 'bs--')
    plt.plot(xeval, yeval_dd, 'c.--')
    plt.plot(xeval, yeval_mm, 'y--')
    plt.legend()

    plt.figure()
    err_l = abs(yeval_l - fex)
    err_dd = abs(yeval_dd - fex)
    err_mm = abs(yeval_mm - fex)
    plt.semilogy(xeval, err_l, 'ro--', label='lagrange')
    plt.semilogy(xeval, err_dd, 'bs--', label='Newton DD')
    plt.semilogy(xeval, err_mm, 'c.--', label='Monomial')
    plt.legend()
    plt.show()


def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N + 1)

    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N + 1):
        yeval = yeval + yint[jj] * lj[jj]

    return yeval


def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]))
    return y


def evalDDpoly(xval, xint, y, N):
    ptmp = np.zeros(N + 1)

    ptmp[0] = 1.
    for j in range(N):
        ptmp[j + 1] = ptmp[j] * (xval - xint[j])

    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N + 1):
        yeval = yeval + y[0][j] * ptmp[j]

    return yeval


def eval_monomial(x, y, z):
    V = np.vander(x, increasing=True)
    a = LA.solve(V, y)

    p_z = 0
    for i in range(len(a)):
        p_z += a[i] * z ** i

    return p_z


if __name__ == "__main__":
    driver()


