import numpy as np
import const
import mle
import inversion
import const


def measure_row(rho: np.array, ax: int, N: int):
    '''
    Simulates a quantum measurement for the same axis in a row.

    :param rho: sample state
    :param ax : measurement axis
        1: x-axis
        2: y-axis
        3: z-axis
    :param N  : number of samples provided
    :return: list of POVM's, list of tuples (axis, measurement result)
    '''
    p = np.trace(np.array([rho@const.edensity[(ax, 0)], rho@const.edensity[(ax, 1)]]), axis1=-2, axis=-1)
    M = np.repeat([const.spovm[ax]], N, axis=0)

    return M, np.array([np.repeat(ax, N), np.random.choice([1, -1], p=p, size=(N,), replace=True)]).T


def measure_once(rho: np.array, ax: int):
    '''
    Simulates a quantum measurement for the same axis in a row.

    :param rho: sample state
    :param ax : measurement axis
        1: x-axis
        2: y-axis
        3: z-axis
    :param N  : number of samples provided
    :return: result of the measurment of N identical states rho and the
        corresponding axis
    '''
    p = np.trace(np.array([rho@const.edensity[(ax, 0)], rho@const.edensity[(ax, 1)]]), axis1=-2, axis=-1)
    M = const.spovm[ax]

    return np.array([ax, np.random.choice([1, -1], p=p)])


def measure_multiple(rho: np.array, axes: np.array, N: int):
    '''
    Simulates several quantum measurements for differents axes.

    :param rho : sample state
    :param axes: measurement axis
        1: x-axis
        2: y-axis
        3: z-axis
    :param N   : number of samples provided
    :return: result of the measurment of N identical states rho and the
        corresponding axis
    '''
    D = np.array([])
    M = np.array([])
    for ax in axes:
        m, d = measure_once(rho, ax)
        D = np.append(D, d)
        M = np.append(M, m)

    return M, D


def recons(D: np.array, method='likelihood', iter=100, M=None):
    '''
    Reconstruct the state according to measuremnt and stated method.

    :param D     : mesurement result, datatype: [axis, eigenvalue]
    :param method: reconstruction method
        likelihood: reconstruction according to maximum likelihood estimator
        inversion : reconstruction according to linear inversion
    :param iter  : number of iteration needed for the maximum likelihood method
    :return: reconstructed state
    '''
    if method!='likelihood' and method!='inversion':
        raise ValueError('Inappropriate value for method. Chose either "likelihood" or "inversion"!')
    elif method=='likelihood':
        return mle.iterative(D, iter)
    elif method=='inversion':
        if M==None:
            raise ValueError("Inappropriate value for the POVM's. If method linear inversion is chosen you have to specify the POVM's.")
        return inversion.linear(M, D)
