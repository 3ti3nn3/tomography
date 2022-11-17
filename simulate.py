import numpy as np
import const
import mle
import inversion
import const


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
