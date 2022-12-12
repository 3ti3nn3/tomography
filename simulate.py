import numpy as np
import const
import mle
import inversion
import const


def measure(rho: np.array, N: int, M: np.array):
    '''
    Simulates quantum measurements for set POVMs.

    :param rho: dxd array of density matrix
    :param N  : sample size
    :param M  : Nxdxd array of POVM set
    :return: N array of measured results
    '''
    p_cum = np.cumsum(np.trace(rho@M, axis1=-2, axis2=-1))
    r     = np.random.uniform(size=N)

    return np.argmax(p_cum>r[:, None], axis=1)
