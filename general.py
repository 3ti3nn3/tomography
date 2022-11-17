import numpy as np
import numpy.linalg as LA
import qutip as qt
import const
import check


def H(rho: np.array):
    '''
    Calculates the conjugate transpose of an array.

    :param rho: Nxdxd or dxd array of density matrices
    :return: complex conjugate
    '''
    if len(rho.shape)==2:
        return np.transpose(np.conjugate(rho))
    elif len(rho.shape)==3:
        return np.transpose(np.conjugate(rho), axes=[0, 2, 1])
    else:
        raise ValueError('Unexpected shape of rho in "general.H" encountered.')


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


def euclidean_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the euclidean distance between the Bloch vector of the given states.

    :param rho_1: density representation of the first state
    :param rho_2: density representation of the second state
    :return: euclidean distance of the Bloch vectors
    '''
    bloch_1 = expect_xyz(rho_1)
    bloch_2 = expect_xyz(rho_2)

    return np.sqrt(np.sum((bloch_1-bloch_2)**2))


def hilbert_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho_1: density representation of the first states
    :param rho_2: density representation of the second state
    :return: Hilbert-Schmidt distance
    '''
    return np.real(np.trace((rho_1-rho_2)**2))


def bures_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Bures distance of the given states according to qutip.

    :param rho_1: density representation of the first state
    :param rho_2: density represnetation of the second state
    :return: fidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return qt.bures_dist(Qrho_1, Qrho_2)


def infidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the fidelity of two given qubit state according to Wikipedia.

    :param rho_1: density representation of the first state
    :param rho_2: density representation of the second state
    :retur: fidelity
    '''
    return np.real(1 - np.trace(rho_1@rho_2) - 2*np.sqrt(LA.det(rho_1)*LA.det(rho_2)))


def realign_povm(M: np.array, phi: np.float, theta: np.float, mirror=True):
    '''
    Rotates the set of POVM by the given angles.

    :param M    : set of POVMs
    :param phi  : polar angle
    :param theta: angular angle
    :return: realigned POVM
    '''
    if mirror:
        R = const.Rz(np.array([-phi-np.pi]))@const.Ry(np.array([np.pi-theta]))
        return R@M@H(R)
    else:
        R = const.Rz(np.array([-phi]))@const.Ry(np.array([theta]))
        return R@M@H(R)


def extract_param(rho: np.array):
    '''
    Determines the angles of rho's orientation and the distance r.

    :param rho: density representation of state
    :return: for pure states: (phi, theta)
        for mixed states: (r, phi, theta)
    '''
    n = expect_xyz(rho)

    r     = np.sqrt(np.sum(n**2))
    phi   = np.arctan2(n[1], n[0])
    theta = np.arccos(n[2]/r)

    return r, phi, theta


def purity(rhos: np.array, prec=1e-15):
    '''
    Computes purity of a given array of density matrix.

    :param rhos: array of density matrices
    :param prec: precision of the purity comparison
    :return: purity
    '''
    return np.trace(rhos@rhos, axis1=-2, axis2=-1, dtype=complex)


def state(rho: np.array, prec=1e-15):
    '''
    Computes whether a given state is actually a valid state.

    :param rho : state in density representation
    :param prec: precision of comparisons
    :return: values
    '''
    return LA.eig(rho)[0], np.trace(rho), np.sum(np.abs(rho-H(rho)))
