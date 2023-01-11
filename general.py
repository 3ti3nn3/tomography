import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
import qutip as qt
import const
import check


# general tools
def H(rho: np.array):
    '''
    Calculates the conjugate transpose of an array of states or one single state.

    :param rho: Nxdxd or dxd array of density matrices
    :return: complex conjugate of density matrices
    '''
    if len(rho.shape)==2:
        return np.transpose(np.conjugate(rho))
    elif len(rho.shape)==3:
        return np.transpose(np.conjugate(rho), axes=[0, 2, 1])
    else:
        raise ValueError(f"Unexpected shape ({rho.shape}) of rho in 'general.H' encountered.")


def purity(rho: np.array):
    '''
    Calculates the purity of an array of density matrices.

    :param rho: Nxdxd or dxd array of density matrices
    :return: purity
    '''
    return np.trace(rho@rho, axis1=-2, axis2=-1)


def expect(operator: np.array, rho: np.array):
    '''
    Calculates the expectation value of an operator and a state.

    :param operator: operator or set of operators
    :param rho     : dxd array of density matrix
    :return: expectation value
    '''
    return np.real(np.trace(operator@rho, axis1=-2, axis2=-1))


def expect_xyz(rho: np.array):
    '''
    Determines the sigma_x, sigma_y and sigma_z expectation value.

    :param rho: Nxdxd or dxd array of density matrices
    :return: Nx3 or 3 array of the three expectation values
    '''
    try:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)]).T
    except:
        return np.real([expect(const.sx, rho), expect(const.sy, rho), expect(const.sz, rho)])


def count(D: np.array, n: np.array):
    '''
    Iterative counter for linear inversion. Also able to process single measurement
    as well as batches of measurement as long as they are provided as an array.

    :param D: data measurement
        dataype: D[i] = [index of POVM]
    :param n: N array of counted POVMs
    :result: updated n
    '''
    for i in range(len(n)):
        n[i] += np.sum(D==i)

    return n


def count_unique(D: np.array, n: np.array):
    '''
    Iterative counter for linear inversion. Also able to process single measurement
    as well as batches of measurement as long as they are provided as an array.

    :param D: N array of data measurement
        dataype: D[i] = [index of POVM]
    :param n: N array of counted POVMs
    :result: updated n
    '''
    unique, count = np.unique(D, return_counts=True)
    n[unique] += count

    return n


def partial_trace(rho: np.array, idx_qubit):
    '''
    Computes the partial trace of the idx_qubit qubit.

    :param rho      : Nxdxd array of states
    :param idx_qubit: list of indices or integer of index of qubit which will be traced out
    return: Nx(d-2)x(d-2) array of reduced states
    '''
    n_qubits = int(np.log2(rho.shape[-1]))
    if type(idx_qubit)==int:
        idx_qubit = [n_qubits-1-idx_qubit]
    elif type(idx_qubit)==list or type(idx_qubit.tolist())==list:
        idx_qubit = np.flip(np.sort(n_qubits-1-np.array(idx_qubit)))
    else:
        raise ValueError(f"Unexpected type of idx_qubit: {type(idx_qubit)}. Choose integer, list or array.")
    assert len(idx_qubit)==0 or idx_qubit[0]<n_qubits, 'Index of qubit which should be reduced exceeds the number of qubits.'

    t1 = tuple(int(2) for i in range(2*n_qubits))
    t2 = (int(2)**(n_qubits-len(idx_qubit)), int(2)**(n_qubits-len(idx_qubit)))

    if len(rho.shape)==3:
        N   = rho.shape[0]
        rho = rho.reshape(N, *t1)

        for i in idx_qubit:
            rho = np.trace(rho, axis1=i+1, axis2=i+n_qubits+1)
            n_qubits -= 1
        return rho.reshape(N, *t2)
    else:
        rho = rho.reshape(*t1)

        for i in idx_qubit:
            rho = np.trace(rho, axis1=i, axis2=i+n_qubits)
            n_qubits -= 1
        return rho.reshape(*t2)


def tensorproduct_axis_1(rhos: np.array, dim1: int, dim2: int):
    rho1, rho2 = np.split(rhos, [dim1**2])
    return np.kron(rho2.reshape(dim2, dim2), rho1.reshape(dim1, dim1))

def tensorproduct_axis_2(rhos_1: np.array, rhos_2: np.array, dim1: int):
    return np.kron(rhos_2, rhos_1.reshape(dim1, dim1))

def tensorproduct_axis_3(rhos_2: np.array, rhos_1: np.array, dim2: int):
    return np.kron(rhos_2.reshape(dim2, dim2), rhos_1)

def tensorproduct(rhos_1: np.array, rhos_2: np.array):
    '''
    Computes the tensorproduct of two arrays of same lenght: rho_1 otimes rho_2.

    :param rhos_1: Mxdxd array of states
    :param rhos_2: Nxdxx array of states
    :return: Nxd^2xd^2 or Mxd^2xd^2 array of states
    '''
    if len(rhos_1.shape)==2 and len(rhos_2.shape)==2:
        return np.kron(rhos_2, rhos_1)

    elif len(rhos_1.shape)==3 and len(rhos_2.shape)==3:
        assert rhos_1.shape[0]==rhos_2.shape[0], f"Unexpected shape: {rhos_1.shape}, {rhos_2.shape}"
        n1, dim1,_ = rhos_1.shape
        n2, dim2,_ = rhos_2.shape
        rhos = np.concatenate((rhos_1.reshape(n1, -1), rhos_2.reshape(n2, -1)), axis=1)
        return np.apply_along_axis(tensorproduct_axis_1, 1, rhos, dim1, dim2)

    elif len(rhos_1.shape)==3 and len(rhos_2.shape)==2:
        n1, dim1, _ = rhos_1.shape
        return np.apply_along_axis(tensorproduct_axis_2, 1, rhos_1.reshape(n1, -1), rhos_2, dim1)

    elif len(rhos_1.shape)==2 and len(rhos_2.shape)==3:
        n2, dim2, _ = rhos_2.shape
        return np.apply_along_axis(tensorproduct_axis_3, 1, rhos_2.reshape(n2, -1), rhos_1, dim2)

    else:
        raise ValueError(f"Input has unexpected shape: {rhos_1.shape}, {rhos_2.shape}")


def tensorproduct_cum(rhos: np.array):
    '''
    :param rhos: Nxdxd array of states
    :return: d^Nxd^N array of state
    '''
    N   = rhos.shape[0]
    rho = rhos[0]

    for i in range(1, N):
        rho = tensorproduct(rho, rhos[i])

    return rho


# rotation matrices
def Rx(alpha):
    '''
    Calculates rotation matrix about x-axis.

    :param alpha: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.cos(alpha/2), -1j*np.sin(alpha/2)],
                         [-1j*np.sin(alpha/2), np.cos(alpha/2)]], axes=[2, 0, 1])


def Ry(theta):
    '''
    Calculates rotation matrix about x-axis.

    :param theta: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], axes=[2, 0, 1])


def Rz(phi):
    '''
    Calculates rotation matrix about x-axis.

    :param phi: rotation angle
    :return: rotation matrix
    '''
    return np.transpose([[np.exp(1j*phi/2, dtype=np.complex), np.zeros(len(phi), dtype=np.complex)],
                         [np.zeros(len(phi), dtype=np.complex), np.exp(-1j*phi/2, dtype=np.complex)]], axes=[2, 0, 1])


def R(n: np.array, alpha: float):
    '''
    Calculates rotation matrix about given unit vector n.

    :param n    : 1x3 array of unit vector
    :param theta: angle
    :return 2x2 array of rotation matrix
    '''
    return np.cos(alpha/2)*np.eye(2) - 1j*np.sin(alpha/2)* np.sum(n[:, None, None]*const.sa, axis=0)


def n(phi: float, theta: float):
    '''
    Returns unit vector on the sphere.

    :param phi  : polar angle
    :param theta: azimuth angle
    :return: 1x3 array of unit vector
    '''
    return np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])


def R_product(phi_1: float, theta_1: float, alpha_1: float, phi_2: float, theta_2: float, alpha_2: float):
    '''
    Returns rotation matrix.

    :param angles_1: angles for first rotation matrix
    :param angles_2: angles for second rotation matrix
    :return: 4x4 array of product rotation matrix
    '''
    return tensorproduct(R(n(phi_1, theta_1), alpha_1), R(n(phi_2, theta_2), alpha_2))


# distance measures
def euclidean_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the euclidean distance.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :return: euclidean distance of the Bloch vectors
    '''
    bloch_1 = expect_xyz(rho_1)
    bloch_2 = expect_xyz(rho_2)

    return np.sqrt(np.sum((bloch_1-bloch_2)**2))


def hilbert_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Hilbert-Schmidt distance.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :return: Hilbert-Schmidt distance
    '''
    return np.real(np.trace((rho_1-rho_2)**2))


def bures_dist(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the Bures distance of the given states according to qutip.

    :param rho_1: dxd array of density marix
    :param rho_2: dxd array of density matrix
    :return: fidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return qt.bures_dist(Qrho_1, Qrho_2)


def infidelity(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the infidelity of two one qubit states according to Wikipedia.

    :param rho_1: dxd array of density matrix
    :param rho_2: dxd array of density matrix
    :retur: infidelity
    '''
    if np.any(check.purity(np.array([rho_1, rho_2]))):
        return 1-np.real(np.trace(rho_1@rho_2))
    elif rho_1.shape[-1]==2:
        return 1-np.real(np.trace(rho_1@rho_2) + 2*np.sqrt(LA.det(rho_1)*LA.det(rho_2)))
    else:
        return 1-np.real(np.trace(sqrtm(rho_1@rho_2))**2)


# N0-step representations
def N_exp(N_max: int, alpha: float):
    '''
    Calculates the exponential representation of N0 in two step adaptive scheme.

    :param N_max: N_max
    :param alpha: exponent
    :return: N0
    '''
    return int(N_max**alpha)


def N_frac(N_max: int, alpha: float):
    '''
    Calculates the fractional representation of N0. in the two step adaptive scheme.

    :param N_max: N_max
    :param alpha: fraction
    :return: N0
    '''
    return int(N_max*alpha)


# product sampling
def sample_product(dim1: int, dim2: int, N: int, f_sample):
    '''
    Samples product states according to f_sample function.

    :param dim_1   : dimension of the first product state
    :param dim_2   : dimension of the second product state
    :param N       : number of states
    :param f_sample: function from which the single product states are sampled from
    :return: Nxdxd array of product states
    '''
    rhos_1, rhos_2 = f_sample(dim1, N), f_sample(dim2, N)

    return tensorproduct(rhos_1, rhos_2)


# product measurements
def pauli4(dim: int):
    '''
    Pauli-4 POVM for arbitrary dimension.

    :param dim: dimension
    :return: Nxdxd array of POVM set
    '''
    assert np.log2(dim)%1==0, 'Dimension is not a power of 2.'
    n_qubits = int(np.log2(dim))

    idx = np.meshgrid(*np.repeat(np.arange(4)[None,], n_qubits, axis=0))
    M   = const.pauli4[idx[0].flatten()]
    for i in range(1, n_qubits):
        M = tensorproduct(M, const.pauli4[idx[i].flatten()])
    return M


def pauli6(dim: int):
    '''
    Pauli-6 POVM for arbitrary dimension.

    :param dim: dimension
    :return: Nxdxd array of POVM set
    '''
    assert np.log2(dim)%1==0, 'Dimension is not a power of 2.'
    n_qubits = int(np.log2(dim))

    idx = np.meshgrid(*np.repeat(np.arange(6)[None,], n_qubits, axis=0))
    M   = const.pauli6[idx[0].flatten()]
    for i in range(1, n_qubits):
        M = tensorproduct(M, const.pauli6[idx[i].flatten()])
    return M


def sic(dim: int):
    '''
    Pauli-6 POVM for arbitrary dimension.

    :param dim: dimension
    :return: Nxdxd array of POVM set
    '''
    assert np.log2(dim)%1==0, 'Dimension is not a power of 2.'
    n_qubits = int(np.log2(dim))

    idx = np.meshgrid(*np.repeat(np.arange(4)[None,], n_qubits, axis=0))
    M   = const.sic[idx[0].flatten()]
    for i in range(1, n_qubits):
        M = tensorproduct(M, const.sic[idx[i].flatten()])
    return M


povm = {}
povm['Pauli-4']  = pauli4
povm['Pauli-6']  = pauli6
povm['SIC-POVM'] = sic
