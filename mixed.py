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
    rho = np.multiply(1/np.trace(B, axis1=-2, axis2=-1)[:, None, None].repeat(dim, axis=1).repeat(dim, axis=2), B)

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
        return np.multiply(1/np.trace(rho, axis1=-2, axis2=-1)[:, None, None].repeat(dim, axis=1).repeat(dim, axis=2), rho)


def sample_product_hilbert(dim: int, N: int):
    '''
    Samples product states using sample_hilbert. If dimension correspond to an even number
    of qubits, a symmetric product state is generated. Elsewise a prodct state of one qubit
    and the remaining qubits is generated.

    :param dim: dimension
    :param N  : number of states
    :return: Nxdxd array of product states
    '''
    if np.log2(dim)%1==0:
        return general.sample_product(int(np.sqrt(dim)), int(np.sqrt(dim)), N, sample_hilbert)
    else:
        return general.sample_product(2, int(np.sqrt(dim/2)), N, sample_hilbert)


def sample_product_bures(dim: int, N: int):
    '''
    Samples product states using sample_hilbert. If dimension correspond to an even number
    of qubits, a symmetric product state is generated. Elsewise a prodct state of one qubit
    and the remaining qubits is generated.

    :param dim: dimension
    :param N  : number of states
    :return: Nxdxd array of product states
    '''
    if np.log2(dim)%1==0:
        return general.sample_product(int(np.sqrt(dim)), int(np.sqrt(dim)), N, sample_bures)
    else:
        return general.sample_product(2, int(np.sqrt(dim/2)), N, sample_bures)
