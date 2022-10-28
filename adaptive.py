import numpy as np

import simulate
import general
import inversion

def two_step(rho: np.array, M0: np.array, N: int, a=0.5):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho: true state
    :param M0: initial POVM set
    :param N  : total number of measurements
    :param a  : this hyperparameter determines the amount of measurments without realignment
    :return: adaptive estimated state
    '''
    N0    = int(N**a)
    D0    = simulate.measure(rho, N0, M0)
    rho_0 = inversion.linear(D0, M0)

    _, phi, theta = general.extract_param(rho_0)
    M1    = general.realign_povm(M0, phi, theta)
    N1    = int(N-N0)
    D1    = simulate.measure(rho, N1, M1)
    rho_1 = inversion.linear(D1, M1)

    return 1/N * (N0*rho_0 + N1*rho_1)
