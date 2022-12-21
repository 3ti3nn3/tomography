import numpy as np
import numpy.linalg as LA
import general


def R_povm(phi: np.float, theta: np.float):
    '''
    Determines the rotation matrix for a rotation on the block sphere for the given angles.

    :param phi  : polar angle
    :param theta: azimutal angle
    :return: 2x2 array of ration matrix
    '''
    return general.Rz(np.array([-phi]))@general.Ry(np.array([theta]))


def R_povm_mirror(phi: np.float, theta: np.float):
    '''
    Determines the rotation matrix for a rotation on the block sphere for the given angles.

    :param phi  : polar angle
    :param theta: azimutal angle
    :return: 2x2 array of ration matrix
    '''
    return general.Rz(np.array([-phi-np.pi]))@general.Ry(np.array([np.pi-theta]))


def extract_param(rho: np.array):
    '''
    Determines the angles of rho's orientation and the distance r.

    :param rho: dxd array of density matrix
    :return: tuple of (r, phi, theta)
    '''
    n = general.expect_xyz(rho)

    r     = np.sqrt(np.sum(n**2))
    phi   = np.arctan2(n[1], n[0])
    theta = np.arccos(n[2]/r)

    return r, phi, theta


def extract_angles(rho: np.array):
    '''
    Determines the angles of rho's orientation.

    :param rho: 2x2 array of density matrix
    :return: tuple of (phi, theta)
    '''
    n = general.expect_xyz(rho)

    r     = np.sqrt(np.sum(n**2))
    phi   = np.arctan2(n[1], n[0])
    theta = np.arccos(n[2]/r)

    return phi, theta


def rotation(rho: np.array, M: np.array):
    '''
    Rotates the set of POVM in the eigenbasis of rho.

    :param M    : Nxdxd array of set of POVMs
    :param phi  : polar angle
    :param theta: angular angle
    :return: Nxdxd realigned POVMs
    '''
    n_qubits = int(np.log2(rho.shape[-1]))

    if n_qubits==1:
        R = R_povm_mirror(*extract_angles(rho))
    elif n_qubits==2:
        rho_A = general.partial_trace(rho, 1)
        rho_B = general.partial_trace(rho, 0)
        R = general.tensorproduct(R_povm_mirror(*extract_angles(rho_A)), R_povm_mirror(*extract_angles(rho_B)))
    else:
        raise Error(f"More than 2 qubits not allowad.")

    return R@M@general.H(R)


def eigenbasis(rho: np.array, M: np.array):
    '''
    Transforms the POVM in the eigenbasis of rho.

    :param rho   : dxd array of state
    :param M     : Nxdxd array of set of POVMs
    :return: Nxdxd realigned POVMs
    '''
    _, U = LA.eigh(rho)
    return U@M@general.H(U)


def product_eigenbasis(rho: np.array, M: np.array):
    '''
    Transforms the POVM in the eigenbasis of its reduces components.

    :param rho   : dxd array of state
    :param M     : Nxdxd array of set of POVMs
    :return: Nxdxd realigned POVMs
    '''
    n_qubits = int(np.log2(rho.shape[-1]))

    rho_red = np.empty((n_qubits, 2, 2), dtype=np.complex)
    U_red   = np.empty((n_qubits, 2, 2), dtype=np.complex)

    for i in range(n_qubits):
        shadow = np.ones(n_qubits, dtype=bool)
        shadow[i] = False

        rho_red[i]  = general.partial_trace(rho, np.arange(n_qubits)[shadow])
        _, U_red[i] = LA.eigh(rho_red[i])

    U = general.tensorproduct_cum(U_red)

    return U@M@general.H(U)
