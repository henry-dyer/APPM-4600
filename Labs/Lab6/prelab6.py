
# slope of both log plots is between 1, 2 to convergence is super linear

import numpy as np
import matplotlib.pyplot as plt

def forward_difference(f, x, h):
    approx = []
    for step in h:
        deriv = (f(x + step) - f(x)) / step
        approx.append([step, deriv])

    return approx


def centered_difference(f, x, h):
    approx = []
    for step in h:
        deriv = (f(x + step) - f(x - step)) / (2 * step)
        approx.append([step, deriv])

    return approx


f0 = lambda x: np.cos(x)

h0 = 0.01 * 2. ** (-np.arange(0, 10))

x0 = np.pi / 2

forward = (forward_difference(f0, x0, h0))

centered = (centered_difference(f0, x0, h0))

plt.figure()
plt.loglog(forward[0][:], forward[1][:], label='Forward Diff')
plt.loglog(centered[0][:], centered[1][:], label='Centered Diff')
plt.xlabel('log(h)')
plt.ylabel('log(deriv)')
plt.title('Log-Log Plot')
plt.legend()
plt.show()
