import numpy as np
from scipy.stats import unitary_group
import general


def sample_unitary(dim: int, N: int):
    '''
    Samples uniformly distributed states using arbitrary unitary transformations.

    :param dim: dimension
    :param N  : number of states
    :return: Nxdxd array of states
    '''
    rho_0      = np.zeros((dim, dim), dtype=np.complex)
    rho_0[0,0] = 1

    U = unitary_group.rvs(dim, size=N)
    try:
        UH  = np.transpose(np.conjugate(U), axes=[0, 2, 1])
    except:
        UH = np.transpose(np.conjugate(U))

    return U@rho_0@UH


def sample_product_unitary(dim: int, N: int):
    '''
    Samples product states using sample_unitary. If dimension correspond to an even number
    of qubits, a symmetric product state is generated. Elsewise a prodct state of one qubit
    and the remaining qubits is generated.

    :param dim: dimension
    :param N  : number of states
    :return: Nxdxd array of product states
    '''
    if np.log2(dim)%1==0:
        return general.sample_product(int(np.sqrt(dim)), int(np.sqrt(dim)), N, sample_unitary)
    else:
        return general.sample_product(2, int(np.sqrt(dim/2)), N, sample_unitary)
