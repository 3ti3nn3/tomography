import numpy as np
import const
import mle
import inversion
import const


def measure_slow(rho: np.array, N: int):
    '''
    Simulates several quantum measurements for a set of operators.

    :param rho: state to sample from
    :param N  : sample size
    :return: array of N measured results and the corresponding axis
    '''
    axs = np.random.randint(0, high=3, size=N)
    p1  = np.trace([3*rho@const.M_up[ax] for ax in axs], axis1=-2, axis2=-1)

    if np.all(np.imag(p1)<1e-14):
        p1 = np.real(p1)
    else:
        raise ValueError('Contradiction: Complex probabilities!')

    p      = np.array([1-p1, p1]).T
    choice = lambda p: np.random.choice([0, 1], p=p)

    return np.array([axs, np.apply_along_axis(choice, 1, p)]).T


def measure_unefficiently(rho: np.array, N: int):
    '''
    Simulates several quantum measurements for a set of operators.

    :param rho: state to sample from
    :param N  : sample size
    :return: array of N measured results and the corresponding axis
    '''
    axs = np.random.randint(0, high=3, size=N)
    ax, ax_counts = np.unique(axs, return_counts=True)

    p10 = np.real(np.trace(3*rho@const.M_up[ax[0]]))
    p11 = np.real(np.trace(3*rho@const.M_up[ax[1]]))
    p12 = np.real(np.trace(3*rho@const.M_up[ax[2]]))

    D0 = np.array([np.repeat(0, ax_counts[0]), np.random.choice([0, 1], p=[1-p10, p10], size=ax_counts[0])]).T
    D1 = np.array([np.repeat(1, ax_counts[1]), np.random.choice([0, 1], p=[1-p11, p11], size=ax_counts[1])]).T
    D2 = np.array([np.repeat(2, ax_counts[2]), np.random.choice([0, 1], p=[1-p12, p12], size=ax_counts[2])]).T

    D = np.concatenate((D0, D1, D2), axis=0)
    np.random.shuffle(D)

    return D


def measure(rho: np.array, N: int, M: np.array):
    '''
    Simulates several quantum measurements for set of operators.

    :param rho: state to sample from
    :param N  : sample size
    :param M  : POVM set
    :return: array of N measured results with numbers between 0 and len(M)
        sampled according to their probabilities
    '''
    p_cum = np.cumsum(np.trace(rho@M, axis1=-2, axis2=-1))
    r     = np.random.uniform(size=N)

    return np.argmax(p_cum>r[:, None], axis=1)


def recons(D: np.array, method='likelihood', M=const.pauli4, iter=1000):
    '''
    Reconstruct the state according to provided data and stated method.

    :param D     : mesurement data
        datatype: D[i] = [index, spin up measured yes(=1) or no(=0)]
    :param method: reconstruction method
        likelihood: reconstruction according to maximum likelihood estimator
        inversion : reconstruction according to linear inversion
    :param iter  : number of iterations needed for the maximum likelihood method
    :param M     : array of POVMs
    :return: reconstructed state
    '''
    if method=='likelihood':
        return mle.iterative(D, iter)
    elif method=='inversion':
        n = inversion.count(D, np.zeros((len(M), 2)))
        return inversion.linear(n)
    else:
        raise ValueError('Inappropriate value for method. Chose either "likelihood" or "inversion"!')
