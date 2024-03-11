import numpy as np

def eval_line(points, alpha):

    if points[0][0] == points[1][0]:
        print('Points with same x - coordinate given, exiting')
        return
    else:
        m = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        b = points[0][1] - m * points[0][0]

    return m * alpha + b


if __name__ == '__main__':
    points = np.array([[0, 1], [1, 2]])

    alpha = 3

    print(eval_line(points, alpha))