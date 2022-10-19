import numpy as np


def count(D: np.array, n: np.array):
    '''
    Iterative counter for linear inversion.

    :param D: datatype: array of [axes, measured eigenvalue]
    :param n: list with number of spin up results and number of spin down reults
        for corresponding POVM
    :result: updated n
    '''
    J = len(n)
    I = np.concatenate(np.transpose(np.meshgrid(np.arange(J), [0, 1]), axes=[2, 1, 0]))

    for i in I:
        n[tuple(i)] += np.sum(np.all(D==i, axis=1))

    return n


def linear(n: np.array, M: np.array):
    '''
    Calculates the best estimator according to the theory of linear inversion.

    :param n: list with number of spin up results and number of spin down reults
        for corresponding POVM
    :param M: set of POVM
    :return: the linear inversion estimator
    '''
    N     = np.sum(n, axis=1)
    p     = (n[:,1] - n[:,0])/N
    M     = np.array(M)
    T     = np.einsum('alk,bkl->ab', M, M)
    T_inv = np.linalg.inv(T)

    return np.einsum('i,ji,jnk->nk', p, T_inv, M)
