import numpy as np

def expect(operator: np.array, rho: np.array):
    '''
    Calculates the expectation value of an operator and a state.

    :param operator: operator
    :param rho     : state
    :return: expectation value
    '''
    return np.trace(operator@rho, axis1=-2, axis2=-1)


def hilbert_schmidt_distance(rho1: np.array, rho2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho1: density representation of the first states
    :param rho2: density representation of the second state
    :return: Hilbert-Schmidt distance
    '''
    return np.trace((rho1-rho2)**2)
