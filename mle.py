import numpy as np
import const


def iterative_iter(rho: np.array, M_D: np.array):
    '''
    Performs one step of iterative MLE according to given measurement.

    :param rho: current numerical approximation of the estimator
    :param M_D: array of eigenstate matrices
    :return: updated estimator
    '''
    p = np.einsum('ik,nki->n', rho, M_D)
    R = np.einsum('n,nij->ij', 1/p, M_D)
    update = R@rho@R

    return update/np.trace(update)


def iterative(D: np.array, M: np.array, iter=500):
    '''
    Performs iterative MLE.

    :param D   : measured data set
        datatype: D[i] = [index of POVM]
    :param M   : set of POVMs
    :param iter: number of iterations
    :return: iterative MLE estimator
    '''
    M_D = M[D]
    rho = np.eye(2)/2

    for j in range(iter):
        rho = iterative_iter(rho, M_D)

    return rho
