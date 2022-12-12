import numpy as np
import numpy.linalg as LA
import qutip as qt
import const
import check


# rotation matrices
def Rx(alpha):
    '''
    Calculates rotation matrix about x-axis.

    :param alpha: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.cos(alpha/2), -1j*np.sin(alpha/2)],
                         [-1j*np.sin(alpha/2), np.cos(alpha/2)]], axes=[2, 0, 1])


def Ry(theta):
    '''
    Calculates rotation matrix about x-axis.

    :param theta: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], axes=[2, 0, 1])


def Rz(phi):
    '''
    Calculates rotation matrix about x-axis.

    :param phi: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.exp(1j*phi/2, dtype=np.complex), np.zeros(len(phi), dtype=np.complex)],
                         [np.zeros(len(phi), dtype=np.complex), np.exp(-1j*phi/2, dtype=np.complex)]], axes=[2, 0, 1])


def H(rho: np.array):
    '''
    Calculates the conjugate transpose of an array of states or one single state.

    :param rho: Nxdxd or dxd array of density matrices
    :return: complex conjugate of density matrices
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

    :param operator: operator or set of operators
    :param rho     : dxd array of density matrix
    :return: expectation value
    '''
    return np.real(np.trace(operator@rho, axis1=-2, axis2=-1))


def expect_xyz(rho: np.array):
    '''
    Determines the sigma_x, sigma_y and sigma_z expectation value.

    :param rho: Nxdxd or dxd array of density matrices
    :return: Nx3 or 3 array of the three expectation values
    '''
    try:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)]).T
    except:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)])


def euclidean_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the euclidean distance.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :return: euclidean distance of the Bloch vectors
    '''
    bloch_1 = expect_xyz(rho_1)
    bloch_2 = expect_xyz(rho_2)

    return np.sqrt(np.sum((bloch_1-bloch_2)**2))


def hilbert_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :return: Hilbert-Schmidt distance
    '''
    return np.real(np.trace((rho_1-rho_2)**2))


def bures_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Bures distance of the given states according to qutip.

    :param rho_1: dxd array of density marix
    :param rho_2: dxd array of density matrix
    :return: fidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return qt.bures_dist(Qrho_1, Qrho_2)


def infidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the infidelity of two one qubit states according to Wikipedia.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :retur: infidelity
    '''
    return np.real(1 - np.trace(rho_1@rho_2) - 2*np.sqrt(LA.det(rho_1)*LA.det(rho_2)))


def realign_povm(M: np.array, phi: np.float, theta: np.float, mirror=True):
    '''
    Rotates the set of POVM by the given angles.

    :param M    : Nxdxd array of set of POVMs
    :param phi  : polar angle
    :param theta: angular angle
    :return: Nxdxd realigned POVMs
    '''
    if mirror:
        R = Rz(np.array([-phi-np.pi]))@Ry(np.array([np.pi-theta]))
        return R@M@H(R)
    else:
        R = Rz(np.array([-phi]))@Ry(np.array([theta]))
        return R@M@H(R)


def extract_param(rho: np.array):
    '''
    Determines the angles of rho's orientation and the distance r.

    :param rho: dxd array of density matrix
    :return: tuple of (r, phi, theta)
    '''
    n = expect_xyz(rho)

    r     = np.sqrt(np.sum(n**2))
    phi   = np.arctan2(n[1], n[0])
    theta = np.arccos(n[2]/r)

    return r, phi, theta


def purity(rho: np.array):
    '''
    Calculates the purity of an array of density matrices.

    :param rho: Nxdxd or dxd array of density matrices
    :return: purity
    '''
    return np.trace(rho@rho, axis1=-2, axis2=-1, dtype=complex)


def N_exp(N_max: int, alpha: float):
    '''
    Calculates the exponential representation of N0 in two step adaptive scheme.

    :param N_max: N_max
    :param alpha: exponent
    :return: N0
    '''
    return int(N_max**alpha)


def N_frac(N_max: int, alpha: float):
    '''
    Calculates the fractional representation of N0. in the two step adaptive scheme.

    :param N_max: N_max
    :param alpha: fraction
    :return: N0
    '''
    return int(N_max*alpha)
