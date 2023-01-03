import numpy as np
import cvxpy as cp
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
    n   = general.count(D, np.zeros(len(M), dtype=np.int))

    iter_max = 500
    dist     = float(1)

    rho_1 = np.eye(dim)/dim
    rho_2 = np.eye(dim)/dim

    j = 0
    while j<iter_max and dist>1e-14:
        p      = np.einsum('ik,nki->n', rho_1, M)
        R      = np.einsum('n,n,nij->ij', n, 1/p, M)
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
    N = len(D)
    dim = M.shape[-1]

    # build likelihood
    n = general.count(D, np.zeros(N, dtype=np.int))
    M_D = M[D]

    # initialise gradient descent
    rho = cp.Variable((dim, dim), hermitian=True)
    constraints = [rho>>0, cp.real(cp.trace(rho))==1]

    # loglike = cp.sum( [cp.prod([n[i], cp.log(cp.real(cp.trace(cp.matmul(M[i], rho))))]) for i in range(N)] )
    # loglike = cp.real(cp.trace(rho))
    loglike = cp.sum( [cp.log(cp.real(cp.trace(cp.matmul(M_D[i], rho)))) for i in range(6)] )
    prob    = cp.Problem(cp.Maximize(loglike), constraints)

    # solve
    prob.solve()
    return rho.value



def two_step(rho: np.array, M0: np.array, N: int, N0: int, f_align, cup=True, prec=1e-14):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho    : dxd array of density matrix
    :param M0     : Nxdxd array of POVM set
    :param N      : total number of measurements
    :param N0     : number of measurements in the first step
    :param f_align: function about how to align after firs step
    :param cup    : all data or only data after first estimate
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
        M1 = f_align(rho_0, M0)

        # true state
        N1 = int(N-N0)
        D1 = simulate.measure(rho, N1, M1)
        if cup:
            D = np.concatenate([D0, D1], axis=0)
        else:
            D  = D1

        return iterative(D, M1)
