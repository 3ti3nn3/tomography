import numpy as np
import qutip as qt
import const


def expect(operator: np.array, rho: np.array):
    '''
    Calculates the expectation value of an operator and a state.

    :param operator: operator
    :param rho     : state
    :return: expectation value
    '''
    return np.real(np.trace(operator@rho, axis1=-2, axis2=-1))


def expect_xyz(rho: np.array):
    '''
    Determines the sigma_x, sigma_y and sigma_z expectation value.

    :param rho: state in density matrix representation
    :return: array of the three expectation values
    '''
    try:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)]).T
    except:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)])

def hilbert_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho1: density representation of the first states
    :param rho2: density representation of the second state
    :return: Hilbert-Schmidt distance
    '''
    return np.real(np.trace((rho_1-rho_2)**2))


def bures_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Bures distance of the given states according to qutip.

    :param rho1: density representation of the first state
    :param rho2: density represnetation of the second state
    :return: fidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return qt.bures_dist(Qrho_1, Qrho_2)


def fidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the fidelity of the given states according to qutip.

    :param rho1: density representation of the first state
    :param rho2: density represnetation of the second state
    :return: fidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return qt.fidelity(Qrho_1, Qrho_2)
