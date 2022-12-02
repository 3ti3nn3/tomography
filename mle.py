import numpy as np
import const
import pickle
import general
import simulate


def iterative(D: np.array, M: np.array, dim=2):
    '''
    Performs iterative MLE.

    :param D   : measured data set
        datatype: D[i] = [index of POVM]
    :param M   : set of POVMs
    :param iter: number of iterations
    :param rho : information already known about true state, starting point of iteration
    :return: iterative MLE estimator
    '''
    M_D = M[D]

    iter_max = 500
    dist     = float(1)

    rho_1 = np.eye(dim)/dim
    rho_2 = np.eye(dim)/dim

    j = 0
    while j<iter_max and dist>1e-14:
        p      = np.einsum('ik,nki->n', rho_1, M_D)
        R      = np.einsum('n,nij->ij', 1/p, M_D)
        update = R@rho_1@R
        rho_1  = update/np.trace(update)

        dist  = general.infidelity(rho_1, rho_2)
        rho_2 = rho_1

        j += 1

    return rho_1


def two_step(rho: np.array, M0: np.array, N: int, N0: int, cup=True, mirror=True):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho  : true state
    :param M0   : inital POVM set
    :param N    : total number of measurements
    :param N0   : total number of mearurements in the first step
    :return: adaptive estimated state
    '''
    # initial estimate
    D0    = simulate.measure(rho, np.minimum(N, N0), M0)
    rho_0 = iterative(D0, M0)

    if N<=N0:
        return rho_0
    else:
        # rallignment according to initial estimate
        _, phi, theta = general.extract_param(rho_0)
        M1    = general.realign_povm(M0, phi, theta, mirror=mirror)

        # true state
        N1    = int(N-N0)
        D1    = simulate.measure(rho, N1, M1)

        # final guess using previous data
        M_D0 = M0[D0]
        M_D1 = M1[D1]

        if cup:
            M_D = np.concatenate([M_D0, M_D1], axis=0)
        else:
            M_D  = M_D1

        iter_max = 500
        dist     = float(1)

        rho_10 = np.eye(2)/2
        rho_11 = np.eye(2)/2

        j = 0
        while j<iter_max and dist>1e-14:
            p      = np.einsum('ik,nki->n', rho_11, M_D)
            R      = np.einsum('n,nij->ij', 1/p, M_D)
            update = R@rho_11@R
            rho_11  = update/np.trace(update)

            dist   = general.infidelity(rho_10, rho_11)
            rho_10 = rho_11

            j += 1

        return rho_11
