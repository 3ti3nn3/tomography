import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize, NonlinearConstraint
import cvxpy as cp
import general
import check


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

    :param rho: dxd array of state
    :param M  : Nxdxd array of set of POVMs
    :return: Nxdxd realigned POVMs
    '''
    _, U = LA.eigh(rho)
    return U@M@general.H(U)


def product_eigenbasis(rho: np.array, M: np.array):
    '''
    Transforms the POVM in the eigenbasis of its reduces components.

    :param rho: dxd array of state
    :param M  : Nxdxd array of set of POVMs
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


def approx(rho: np.array, M: np.array):
    '''
    Transform the POVM in nearest possible product state.

    :param rho: dxd array of state
    :param M  : Nxdxd array of set of POVMs
    :return: Nxdxd array of realigned POVMs
    '''
    # assert check.purity(rho), f"Approximation scheme is only valid for pure states. Purity: {general.purity(rho)}"

    # initialize minimize function
    def F(x):
        R  = general.R_product(*x)
        return -np.real((R@rho@general.H(R))[0,0])

    x0   = np.array([1,1/2,1,1,1/2,1])*np.pi
    boun = np.array([(0, 2*np.pi), (0, np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, np.pi), (0, 2*np.pi)])

    results = minimize(F, x0, method='SLSQP', bounds=boun)

    R = general.R_product(*results.x)
    # return results
    return R@M@general.H(R)


def approx_unitary(rho: np.array, M: np.array):
    '''
    Transform the POVM in nearest possible product state.

    :param rho: dxd array of state
    :param M  : Nxdxd array of set of POVMs
    :return: Nxdxd array of realigned POVMs
    '''
    # optimization function
    def F(R):
        R1, R2 = np.split(R, 2)
        R1, R2 = R1.reshape(2, 2), R2.reshape(2, 2)
        R_prod = general.tensorproduct(R1, R2)
        _, U = LA.eigh(rho)
        return -np.real(np.trace((R_prod@rho@general.H(R_prod)@U@rho@general.H(U))))

    _, U = LA.eigh(rho)
    print('U applied to rho:', U.T@rho@general.H(U.T))

    # optimization constraints
    def F_con(R, idx_R, idx_entry):
        R = np.split(R, 2)[idx_R]
        R = R.reshape(2, 2)
        return (R@general.H(R))[idx_entry]

    con1 = lambda R: F_con(R, 0, (0,0))-1
    con2 = lambda R: F_con(R, 0, (0,1))
    con3 = lambda R: F_con(R, 0, (1,1))-1
    con4 = lambda R: F_con(R, 0, (1,0))
    con5 = lambda R: F_con(R, 1, (0,0))-1
    con6 = lambda R: F_con(R, 1, (0,1))
    con7 = lambda R: F_con(R, 1, (1,1))-1
    con8 = lambda R: F_con(R, 1, (1,0))

    cons = ({'type': 'eq', 'fun': con1},
            {'type': 'eq', 'fun': con2},
            {'type': 'eq', 'fun': con3},
            {'type': 'eq', 'fun': con4},
            {'type': 'eq', 'fun': con5},
            {'type': 'eq', 'fun': con6},
            {'type': 'eq', 'fun': con7},
            {'type': 'eq', 'fun': con8})

    # optimize
    x0 = np.array([np.eye(2), np.eye(2)]).flatten()

    res = minimize(F, x0, constraints=cons, method='trust-constr')

    return res
