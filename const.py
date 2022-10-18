import numpy as np


# Pauli matrices
se = np.eye(2)
sx = np.array([[0, 1],
                   [1, 0]])
sy = np.array([[0, 1j],
                   [-1j, 0]])
sz = np.array([[1, 0],
                   [0, -1]])
sa  = np.array([sx, sy, sz])
spovm = np.array([se, sx, sy, sz])


# Pauli eigenstates
estate = {}
estate[(1, +1)] = 1/np.sqrt(2)*np.array([1, 1])
estate[(1, -1)] = 1/np.sqrt(2)*np.array([1, -1])
estate[(2, +1)] = 1/np.sqrt(2)*np.array([1, 1j])
estate[(2, -1)] = 1/np.sqrt(2)*np.array([1, -1j])
estate[(3, +1)] = np.array([1, 0])
estate[(3, -1)] = np.array([0, 1])

# Pauli eigenstates in density representation
edensity = {}
edensity[(1, +1)] = 1/2*np.array([[1, 1], [1, 1])
edensity[(1, -1)] = 1/2*np.array([[1, -1], [-1, 1]])
edensity[(2, +1)] = 1/2*np.array([[1, 1j], [-1j, 1]])
edensity[(2, -1)] = 1/2*np.array([[1, 1j], [-1j, 1]])
edensity[(3, +1)] = np.array([[1, 0], [0, 0]])
edensity[(3, -1)] = np.array([[0, 0], [0, 1]])


# rotation matrices
def Rx(alpha):
    return np.transpose([[np.cos(alpha/2), -1j*np.sin(alpha/2)],
                         [-1j*np.sin(alpha/2), np.cos(alpha/2)]], axes=[2, 0, 1])
def Ry(theta):
    return np.transpose([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], axes=[2, 0, 1])
def Rz(phi):
    return np.transpose([[np.exp(1j*phi/2, dtype=complex), np.zeros(len(phi), dtype=complex)],
                         [np.zeros(len(phi), dtype=complex), np.exp(-1j*phi/2, dtype=complex)]], axes=[2, 0, 1])
