import numpy as np


def linear(M: np.array, D: np.array):
    '''
    Calculates the best estimator according to the theory of linear inversion.

    :param M: set of POVM
    :param D: len(M)xNx2 dimensional array of measured frequencies for each POVM in M,
        N describes the number of measurements for each basis
    :return: the linear inversion estimator
    '''
    p     = np.mean(D[:, :, 1], axis=1)
    T     = np.einsum('alk,bkl->ab', M, M)
    T_inv = np.linalg.inv(T)

    return np.einsum('i,ji,jnk->nk', p, T_inv, M)
