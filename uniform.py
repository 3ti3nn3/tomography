import numpy as np

def uniform_angles(N: int):
    """
    Creates an array containing the azimuth and polar angle uniformly
    distributed on the block sphere.

    :param N: Denotes the number of uniformly distributed angles.
    :return: Nx2 array of azimuth and polar angle.
    """
    return

def angles_to_state(angles: np.array):
    """
    Transforms angle parameters array to states in computational basis.

    :param angles: Nx2 dimensional array  with two azimuth and polar angle.
    :return: Nx2 dimensional array of states in computational basis.
    """
    return

def state_to_points(states: np.array):
    """
    Creates a visualization of the given states on the Block sphere.

    :param states: Nx2 dimensional array of states in computational basis.
    :return: Visualization.
    """
