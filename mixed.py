import numpy as np
import const


def generate_uniform(N: int):
    '''
    Generates uniformly distributed Bloch parameters for mixed states.

    :param N: number of samples
    :return: tuple of arrays of radii, polar angles and azimuth angles
    '''
    x     = np.random.uniform(0, 1, size=N)
    y     = -1*np.random.uniform(-1, 0, size=N)

    theta = np.arccos(1-2*x)
    phi   = np.random.uniform(-np.pi, np.pi, size=N)
    r     = np.power(y, 1/3)

    return r, phi, theta


def blochvector_to_density(r: np.array, phi: np.array, theta: np.array):
    '''
    Builds mixed states via the Bloch representation.

    :param r    : array of radii
    :param phi  : array of polar angles
    :param theta: array of azimuth angles
    :return: array of mixed states
    '''
    n = r*np.array([np.cos(phi)*np.sin(theta),
                    np.sin(phi)*np.sin(theta),
                    np.cos(theta)])

    if len(n.shape)==1:
        return 1/2 * (const.se + np.einsum('k,klm->lm', n, const.sa))
    else:
        return 1/2 * (const.se + np.einsum('kn,klm->nlm', n, const.sa))


def hermitian_to_density(dim: int, N: int):
    '''
    Creates random mixed states according to AA^degga/tr(AA^degga).

    :param dim: dimension of the needed random state
    :param N  : number of random states
    :return: array of N uniformly distributed states of dimension dim
    '''
    A  = np.random.normal(size=[N, dim, dim]) + 1j*np.random.normal(size=[N, dim, dim])
    AH = np.transpose(np.conjugate(A), axes=[0, 2, 1])

    B   = A@AH
    rho = np.multiply(1/np.trace(B, axis1=-2, axis2=-1)[:, None, None].repeat(2, axis=1).repeat(2, axis=2), B)

    return rho
