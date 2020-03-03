import numpy as np


def line_eq(x1, y1, x2, y2):
    m = float(y2 - y1) / (x2 - x1)
    q = y1 - (m * x1)

    def _line(x):
        return m * x + q

    return _line


def build_pendence(loss):
    x = []
    ip = []
    r = line_eq(0, loss[0], len(loss) - 1, loss[len(loss) - 1])

    for i in range(len(loss)):
        x.append(i)

    for j in x:
        ip.append(j)

    return ip


def integrals(loss):
    '''
    This function calculates the integral of the original loss function and his slope
    :param loss: original loss function
    :return: integral values of the original loss and slope respectively
    '''
    ip = build_pendence(loss)
    return np.trapz(loss), np.trapz(ip)
