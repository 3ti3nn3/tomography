import numpy as np


def count_unique(D: np.array, n: np.array):
    '''
    Iterative counter for linear inversion. Also able to process single measurement
    as well as batches of measurement as long as they are provided as an array.

    :param D: data measurement
        dataype: D[i] = [index of POVM]
    :param n: list of counted POVMs
    :result: updated n
    '''
    unique, count = np.unique(D, return_counts=True)
    n[unique] += count

    return n


def count(D: np.array, n: np.array):
    '''
    Iterative counter for linear inversion. Also able to process single measurement
    as well as batches of measurement as long as they are provided as an array.

    :param D: data measurement
        dataype: D[i] = [index of POVM]
    :param n: list of counted POVMs
    :result: updated n
    '''
    for i in range(len(n)):
        n[i] += np.sum(D==i)

    return n


def linear(D: np.array, M: np.array):
    '''
    Calculates the best estimator according to the theory of linear inversion.

    :param D: data measurement
        dataype: D[i] = [index of POVM]
    :param M: set of POVM
    :return: the linear inversion estimator
    '''
    N     = len(D)
    n     = count(D, np.zeros(len(M), dtype=int))
    p     = n/N
    T     = np.einsum('alk,bkl->ab', M, M)
    T_inv = np.linalg.inv(T)

    return np.einsum('i,ji,jnk->nk', p, T_inv, M)
