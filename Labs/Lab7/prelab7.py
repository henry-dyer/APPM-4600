import numpy as np
from numpy import linalg as LA

"""
x is vector of independent variables of given points
y is vector of dependent variables of given points
z is the point to evaluate polynomial at after solving vandermonde system
"""


def vandermonde_solve(x, y, z):
    V = np.vander(x, increasing=True)
    a = LA.solve(V, y)

    p_z = 0
    for i in range(len(a)):
        p_z += a[i] * z ** i

    return p_z


"""
Lab Summary:
For this lab we will first be exploring the effectivness of different polynomial interpolation techniques.
We will also be evaluating how the errors behave based on the degree of polynomial we use.
Finally we will explore methods to control the error toward the boundarys (Runge Phenomena) by using more interpolation 
nodes closer to the boundaries.
"""



