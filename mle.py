import numpy as np
import const
import pickle
import general
import simulate


def iterative(D: np.array, M: np.array):
    '''
    Estimates state according to iterative MLE.

    :param D   : N array of measured data
        datatype: D[i] = [index of POVM]
    :param M   : Nxdxd array of POVM set
    :param iter: number of iterations
    :return: dxd array of iterative MLE estimator
    '''
    dim = M.shape[-1]
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

        if j>=40 and j%20==0:
            dist  = general.infidelity(rho_1, rho_2)
        rho_2 = rho_1

        j += 1

    return rho_1


def gradient(D: np.array, M: np.array):
    '''
    Estimates state according to iterative MLE using gradient descent.

    :param D   : N array of measured data
        datatype: D[i] = [index of POVM]
    :param M   : Nxdxd array of POVM set
    :param iter: number of iterations
    :return: dxd array of iterative MLE estimator
    '''
    pass


def two_step(rho: np.array, M0: np.array, N: int, N0: int, cup=True, mirror=True, prec=1e-14):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho   : dxd array of density matrix
    :param M0    : Nxdxd array of POVM set
    :param N     : total number of measurements
    :param N0    : number of measurements in the first step
    :param cup   : all data or only data after first estimate
    :param mirror: align or antialign
    :return: dxd array of adaptive mle estimator
    '''
    # initial constants
    dim = rho.shape[-1]

    # initial estimate
    D0    = simulate.measure(rho, np.minimum(N, N0), M0)
    rho_0 = iterative(D0, M0)

    if N<=N0:
        return rho_0
    else:
        # rallignment according to initial estimate
        M1 = general.realign_povm(rho_0, M0, mirror=mirror)

        # true state
        N1 = int(N-N0)
        D1 = simulate.measure(rho, N1, M1)

        # final guess using previous data
        M_D0 = M0[D0]
        M_D1 = M1[D1]
        if cup:
            M_D = np.concatenate([M_D0, M_D1], axis=0)
        else:
            M_D  = M_D1

        iter_max = 500
        dist     = float(1)
        rho_10   = np.eye(dim)/dim
        rho_11   = np.eye(dim)/dim

        j = 0
        while j<iter_max and dist>1e-14:
            p      = np.einsum('ik,nki->n', rho_11, M_D)
            R      = np.einsum('n,nij->ij', 1/p, M_D)
            update = R@rho_11@R
            rho_11  = update/np.trace(update)

            if j>=40 and j%20==0:
                dist  = general.infidelity(rho_10, rho_11)
            rho_10 = rho_11

            j += 1

        return rho_11
