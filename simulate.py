import numpy as np
import const
import mle
import inversion
import const


def measure(rho: np.array, axes: np.array):
    '''
    Simulates several quantum measurements for a array of operators.

    :param rho : sample state
    (:param M   : the set of POVM on which the measurement is based on)
    :param axes: array of indices of the chosen operator of the set of POVM
    :return: array of the N measured results and the corresponding axis
    '''
    p0 = np.trace([rho@const.edensity[(ax, 0)] for ax in axes], axis1=-2, axis2=-1)
    if np.all(np.imag(p0)<1e-14):
        p0 = np.real(p0)
    else:
        raise ValueError('Contradiction: Complex probabilities!')

    p1 = 1-p0
    p  = np.array([p0, p1]).T

    choice = lambda p: np.random.choice([0, 1], p=p)

    return np.array([axes, np.apply_along_axis(choice, 1, p)]).T


def recons(D: np.array, method='likelihood', iter=1000, M=None):
    '''
    Reconstruct the state according to measuremnt and stated method.

    :param D     : mesurement result, datatype: array of [axis, decoded measured eigenvalue]
        (see "const.py" for decoding scheme)
    :param method: reconstruction method
        likelihood: reconstruction according to maximum likelihood estimator
        inversion : reconstruction according to linear inversion
    :param iter  : number of iteration needed for the maximum likelihood method
    :param M     : array of POVMs
    :return: reconstructed state
    '''
    if method=='likelihood':
        return mle.iterative(D, iter)
    elif method=='inversion':
        if np.any(M==None):
            raise ValueError("Inappropriate value for the POVM's. If method linear inversion is chosen, you have to specify the POVM's.")
        n = inversion.count(D, np.zeros((len(M), 2)))
        return inversion.linear(n, const.spovm)
    else:
        raise ValueError('Inappropriate value for method. Chose either "likelihood" or "inversion"!')
