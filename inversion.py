import numpy as np
import simulate
import general


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


def two_step(rho: np.array, M0: np.array, N: int, N0: int, cup=True, mirror=True):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho  : true state
    :param M0   : initial POVM set
    :param N    : total number of measurements
    :param N0   : total number of measurements in the first step
    :return: adaptive estimated state
    '''
    D0    = simulate.measure(rho, np.minimum(N0, N), M0)
    rho_0 = linear(D0, M0)

    if N<=N0:
        return rho_0
    else:
        _, phi, theta = general.extract_param(rho_0)
        M1    = general.realign_povm(M0, phi, theta, mirror=mirror)
        N1    = int(N-N0)
        D1    = simulate.measure(rho, N1, M1)
        rho_1 = linear(D1, M1)

        if cup:
            return 1/N * (N0*rho_0 + N1*rho_1)
        else:
            return rho_1
