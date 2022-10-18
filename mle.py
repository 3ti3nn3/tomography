import numpy as np
import const


def iterative_one(rho, R_arr):
    '''
    Performs one step of iterative MLE according to given measurement.

    :param rho  : current numerical approximation of the estimator
    :param R_arr: array of eigenstate matrices
    :return: updated estimator
    '''
    p_arr  = np.einsum('ik,nki->n', rho, R_arr)
    R      = np.einsum('n,nij->ij', 1/p_arr, R_arr)
    update = R@rho@R

    return update/np.trace(update)


def iterative(D: list, iter: int):
    '''
    Performs iterative MLE.

    :param D   : measured data set, datayep: [axis, eigenvalue] array of array
    :param iter: number of iterations
    :return: iterative MLE estimator
    '''
    I     = len(D)
    rho   = np.eye(2)/2
    R_arr = np.empty((I, 2, 2), dtype=complex)

    # calculate the eigenstate matrices
    for i in range(I):
        estate   = const.estate[tuple(D[i])]
        R_arr[i] = np.tensordot(estate, np.conjugate(estate), axes=0)

    # perform updates
    for i in range(iter):
        rho = iterative_one(rho, R_arr)

    return rho
