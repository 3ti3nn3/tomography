import numpy as np


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
                            
