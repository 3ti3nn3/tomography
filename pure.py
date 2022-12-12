import numpy as np
from scipy.stats import unitary_group


def sample_unitary(dim: int, N: int):
    '''
    Samples uniformly distributed states using arbitrary unitary transformations.

    :param dim: dimension
    :param N  : number of states
    :return: Nxdxd array of states
    '''
    U   = unitary_group.rvs(dim, size=N)
    try:
        UH  = np.transpose(np.conjugate(U), axes=[0, 2, 1])
    except:
        UH = np.transpose(np.conjugate(U))

    return U@np.array([[1, 0], [0, 0]])@UH
