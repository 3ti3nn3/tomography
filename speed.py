import numpy as np
import time


def measure(func, *args, iterations=10):
    '''
    Measures the speed of the given function.

    :param func      : function which speed should be measured
    :param *args     : arguments of test function
    :param iterations: number of iterations the speed is measured
    :return: time the test function needed
    '''
    t0 = np.empty(iterations, dtype=float)
    t1 = np.empty(iterations, dtype=float)

    try:
        for i in range(iterations):
            t0[i] = time.time()
            func(*args)
            t1[i] = time.time()
    except:
        for i in range(iterations):
            t0[i] = time.time()
            func()
            t1[i] = time.time()

    return np.mean(t1-t0)


def compare(iterations=10, **kwargs):
    '''
    Compares a arbitraty number of functions.

    :param iterations: number of iterations the test function is tested
    :param **kwargs  : dictionary like objekt of the form "name = (func, list of parameters)"
    :return: dictionaries of times each test function needed
    '''
    t_mean = {}

    for key, value in kwargs.items():
        t           = measure(value[0], *value[1], iterations=iterations)
        t_mean[key] = t

    return t_mean
