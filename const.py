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

# Pauli POVM
pauli4 = np.array([M1, M2, M3, np.eye(2)-M1-M2-M3])
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

# Pauli string POVM
string1d = np.array([se/2, sx/2, sy/2, sz/2])

# Bell states
bell = np.empty((4, 4), dtype=np.complex)
bell[0] = 1/np.sqrt(2) * np.array([1, 0, 0, 1])
bell[1] = 1/np.sqrt(2) * np.array([1, 0, 0, -1])
bell[2] = 1/np.sqrt(2) * np.array([0, 1, 1, 0])
bell[3] = 1/np.sqrt(2) * np.array([0, 1, -1, 0])
