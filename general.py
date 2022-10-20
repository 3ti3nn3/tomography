import numpy as np
import const


def expect(operator: np.array, rho: np.array):
    '''
    Calculates the expectation value of an operator and a state.

    :param operator: operator
    :param rho     : state
    :return: expectation value
    '''
    return np.trace(operator@rho, axis1=-2, axis2=-1)


def expect_xyz(rho: np.array):
    '''
    Determines the sigma_x, sigma_y and sigma_z expectation value.

    :param rho: state in density matrix representation
    :return: array of the three expectation values
    '''
    return np.array([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)], dtype=np.float)


def hilbert_schmidt_distance(rho1: np.array, rho2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho1: density representation of the first states
    :param rho2: density representation of the second state
    :return: Hilbert-Schmidt distance
    '''
    return np.trace((rho1-rho2)**2)
