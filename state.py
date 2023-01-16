import numpy as np
import const


def bell1(dim: int, N: int):
    '''
    Returns only bell states.

    :param dim: placeholder
    :param N  : placeholder
    :return: 1x4x4 array of a bell state
    '''
    assert dim==4, 'Bell states are only valid for two qubits.'

    bell = 1/2 * np.array([[[1, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]]])
    return np.repeat(bell, N, axis=0)

def bell2(dim: int, N: int):
    '''
    Returns only bell states.

    :param dim: placeholder
    :param N  : placeholder
    :return: 1x4x4 array of a bell state
    '''
    assert dim==4, 'Bell states are only valid for two qubits.'

    bell = 1/2 * np.array([[[1, 0, 0, -1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-1, 0, 0, -1]]])
    return np.repeat(bell, N, axis=0)


def bell3(dim: int, N: int):
    '''
    Returns only bell states.

    :param dim: placeholder
    :param N  : placeholder
    :return: 1x4x4 array of a bell state
    '''
    assert dim==4, 'Bell states are only valid for two qubits.'

    bell = 1/2 * np.array([[[0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0]]])
    return np.repeat(bell, N, axis=0)


def bell4(dim: int, N: int):
    '''
    Returns only bell states.

    :param dim: placeholder
    :param N  : placeholder
    :return: 1x4x4 array of a bell state
    '''
    assert dim==4, 'Bell states are only valid for two qubits.'

    bell = 1/2 * np.array([[[0, 0, 0, 0],
                            [0, 1, -1, 0],
                            [0, -1, 1, 0],
                            [0, 0, 0, 0]]])
    return np.repeat(bell, N, axis=0)


def maxmixed(dim: int, N: int):
    '''
    Returns the maximally mixed state.

    :param dim: dimension of true state
    :param N  : number of samples
    :return: Nxdxd array of maximally mixed state
    '''
    rho = np.eye(dim)/dim
    return np.repeat(rho[None,:,:], N, axis=0)


def purity(dim: int, N: int):
    '''
    Samples uniformly distributed mixed states with certain purity.
    Only for one qubit possible.

    :param dim: dimension of the state
    :param N  : number of states
    :return: Nxdxd or dxd array of states
    '''
    assert dim==2, f"Purity sampling only valid for one qubit. Dimension of current estimate: {dim}"
    r = 0.75

    x     = np.random.uniform(0, 1, size=N)
    theta = np.arccos(1-2*x)
    phi   = np.random.uniform(-np.pi, np.pi, size=N)

    pauli = np.array([const.se, const.sx, const.sy, const.sz])
    n     = np.array([np.ones(N), r * np.cos(theta)*np.sin(phi), r * np.sin(theta)*np.sin(phi), r * np.cos(phi)]).T

    return 1/2 * np.einsum('ni, ikl->nkl', n, pauli)
