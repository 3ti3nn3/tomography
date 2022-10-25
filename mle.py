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


def iterative_one(D: np.array, M: np.array, iter: int):
    '''
    Performs iterative MLE and stores the result after every new data point.

    :param D   : measured data set
        datatype: D[i] = [index of POVM]
    :param M   : set POVMs
    :param iter: number of iterations
    :return: development of iterative MLE estimator
    '''
    N   = len(D)
    M_D = M[D]
    rho = np.empty((N, 2, 2), dtype=np.complex)

    for i in range(N):
        rho_update = np.eye(2)/2
        for j in range(iter):
            rho_update = iterative_iter(rho_update, M_D[:i])
        rho[i] = rho_update

    return rho
