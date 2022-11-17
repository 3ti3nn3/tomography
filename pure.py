import numpy as np
import qutip as qt
from scipy.stats import unitary_group
import const


def generate_uniform(N: int):
    '''
    Generates data in compliance with transforming surface element. Data is uniformly smapled according to
    the concept of Inverser Transform Sampling.

    :param N: number of uniformly distributed points
    :return: an array both for the polar and the azimuth angle of lenght N
    '''
    x     = np.random.uniform(0, 1, size=N)
    theta = np.arccos(1-2*x)
    phi   = np.random.uniform(-np.pi, np.pi, size=N)

    return phi, theta


def angles_to_density(phi: np.array, theta: np.array):
    '''
    Takes polar and azimuth angles and builds a state in computational basis using
    array representation.

    :param phi  : array of uniformly distributed polar angles
    :param theta: array of uniformly distributed azimuth angles
    :return: array of uniformly distributed states
    '''
    Psi = np.array([np.cos(theta/2), np.sin(theta/2)*np.exp(1j*phi)]).T

    return np.einsum('nk,nj->nkj', Psi, np.conjugate(Psi))


def rotation_to_density(phi, theta):
    '''
    Takes polar and azimuth angles and builds a vector of expecation values in cartesian coordinates.

    :param phi  : array or float of polar angle
    :param theta: array or float of azimuth angle
    return: 3xlen(phi)-dimensional array of expectations values in cartesian coordinates
    '''
    R   = const.Rz(phi)@const.Ry(theta)
    RH  = np.transpose(np.conjugate(R), axes=[0, 2, 1])
    rho = R@np.array([[1, 0], [0, 0]])@RH

    return rho


def unitary_to_density(dim: int, N: int):
    '''
    Creates uniformly distributed states on the Bloch sphere by using arbitrary unitary operators.

    :param dim: dimension
    :param N  : number of random density matrices
    :return: array of N uniformly distributed density matrices
    '''
    U   = unitary_group.rvs(dim, size=N)
    try:
        UH  = np.transpose(np.conjugate(U), axes=[0, 2, 1])
    except:
        UH = np.transpose(np.conjugate(U))
    rho = U@np.array([[1, 0], [0, 0]])@UH

    return rho
