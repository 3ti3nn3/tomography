import numpy as np
import general
from scipy.stats import unitary_group


def sample_hilbert(dim: int, N: int):
    '''
    Samples uniformly distributed mixed states according to Hilbert Schmidt measure.

    :param dim: dimension of the state
    :param N  : number of states
    :return: Nxdxd or dxd array of states
    '''
    A  = np.random.normal(size=[N, dim, dim]) + 1j*np.random.normal(size=[N, dim, dim])
    AH = np.transpose(np.conjugate(A), axes=[0, 2, 1])

    B   = A@AH
    rho = np.multiply(1/np.trace(B, axis1=-2, axis2=-1)[:, None, None].repeat(2, axis=1).repeat(2, axis=2), B)

    if rho.shape[0]==1:
        return rho[0]
    else:
        return rho


def sample_bures(dim: int, N: int):
    '''
    Samples uniformly distributed mixed states according to Bures distance measure.

    :param dim: dimension of the state
    :param N  : number of states
    :return: Nxdxd or dxd array of states
    '''
    A  = np.random.normal(size=[N, dim, dim]) + 1j*np.random.normal(size=[N, dim, dim])
    AH = general.H(A)

    U  = unitary_group.rvs(dim, size=N)
    UH = general.H(U)

    rho = (np.eye(dim)+U) @ A@AH @ (np.eye(dim)+UH)

    if rho.shape[0]==1:
        return rho[0]/np.trace(rho[0])
    else:
        return np.multiply(1/np.trace(rho, axis1=-2, axis2=-1)[:, None, None].repeat(2, axis=1).repeat(2, axis=2), rho)
