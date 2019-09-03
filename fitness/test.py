#!/usr/bin/python3
"""
This is sample fitness method.
"""
import numpy as np


def fitness(x):
    # return sum(abs(x))
    return sum(x ** 2)  # 1st deJong
    return sum(np.cos((x) * 0.5))
    return sum(2*np.sin(x) - (np.cos(x[0]**2) * (0.05 * np.ndim(x)) * x + abs(x[-1])))

    # Himmelblau 3D
    # optimas: 3, 2
    #          -2.805118, 3.131312
    #          -3.779310, -3.283186
    #          3.584428, -1.848126
    return ((x[0] ** 2 + x[1] - 11) ** 1 +
            (x[0] + x[1] ** 2 - 7) ** 1)
