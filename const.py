import numpy as np

# Pauli matrices
se = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
sa  = np.array([sx, sy, sz])

# standard POVM
M1 = 1/6*np.array([[1, 1], [1, 1]])
M4 = 1/6*np.array([[1, -1], [-1, 1]])
M2 = 1/6*np.array([[1, -1j], [1j, 1]])
M5 = 1/6*np.array([[1, 1j], [-1j, 1]])
M3 = 1/3*np.array([[1, 0], [0, 0]])
M6 = 1/3*np.array([[0, 0], [0, 1]])

M_up   = np.array([M1, M2, M3])
M_down = np.array([M4, M5, M6])

# Pauli-4 POVM
pauli4 = np.array([M1, M2, M3, np.eye(2)-M1-M2-M3])

# Pauli-6 POVM
pauli6 = np.array([M1, M4, M2, M5, M3, M6])

# SIC-POVM states
stateSic    = np.empty((4, 2), dtype=np.complex)
stateSic[0] = np.array([1, 0])
stateSic[1] = np.array([np.sqrt(1/3), np.sqrt(2/3)])
stateSic[2] = np.array([np.sqrt(1/3), np.sqrt(2/3)*np.exp(2*np.pi*1j/3)])
stateSic[3] = np.array([np.sqrt(1/3), np.sqrt(2/3)*np.exp(4*np.pi*1j/3)])

# SIC-POVM
sic    = np.empty((4, 2, 2), dtype=np.complex)
sic[0] = 1/2*stateSic[0][:,None]@np.conjugate(stateSic[0][None,:])
sic[1] = 1/2*stateSic[1][:,None]@np.conjugate(stateSic[1][None,:])
sic[2] = 1/2*stateSic[2][:,None]@np.conjugate(stateSic[2][None,:])
sic[3] = 1/2*stateSic[3][:,None]@np.conjugate(stateSic[3][None,:])

# dictionary of all povms
povm = {}
povm['Pauli-4']  = pauli4
povm['Pauli-6']  = pauli6
povm['SIC-POVM'] = sic

# Pauli string POVM
string1d = np.array([se/2, sx/2, sy/2, sz/2])


# rotation matrices
def Rx(alpha):
    return np.transpose([[np.cos(alpha/2), -1j*np.sin(alpha/2)],
                         [-1j*np.sin(alpha/2), np.cos(alpha/2)]], axes=[2, 0, 1])
def Ry(theta):
    return np.transpose([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], axes=[2, 0, 1])
def Rz(phi):
    return np.transpose([[np.exp(1j*phi/2, dtype=np.complex), np.zeros(len(phi), dtype=np.complex)],
                         [np.zeros(len(phi), dtype=np.complex), np.exp(-1j*phi/2, dtype=np.complex)]], axes=[2, 0, 1])
