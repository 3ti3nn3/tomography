import numpy as np
import numpy.linalg as LA
import general


def state(rho: np.array, prec=1e-14):
    '''
    Checks whether a given state is actually a valid state.

    :param rho : state in density representation
    :param prec: precision of comparisons
    :return: boolean
    '''
    bool_pos   = np.all(LA.eig(rho)[0]>=-prec)
    bool_trace = np.abs(np.trace(rho)-1<prec)
    bool_herm  = np.sum(np.abs(rho-general.H(rho)))<prec

    return bool_pos and bool_trace and bool_herm


def purity(rhos: np.array, prec=1e-15):
    '''
    Check purity of a given array of density matrix.

    :param rhos: array of density matrices
    :param prec: precision of the purity comparison
    :return: boolean whether given density matrices are pure or not
    '''
    # compute purity
    purity = np.trace(rhos@rhos, axis1=-2, axis2=-1, dtype=complex)

    # exclude inaccuracies caused by finte number representation of a computers
    if np.all(np.abs(np.imag(purity)) < prec) and np.all(np.abs(purity-1) < prec):
        return True
    else:
        shadow = np.logical_or(np.all(np.abs(np.imag(purity)) >= prec), np.all(np.abs(purity-1) >= prec))
        return False


def povm(M: np.array, prec=1e-15):
    '''
    Checks a given set of matrices whether they are a POVM or not.

    :param M   : Nxdxd dimensional array where d is the dimension of the considered system
    :param prec: precision of comparisons
    :return: boolean wheter the criteria of a POVM are fulfilled by the given set
    '''
    # hermitian
    bool_herm = M==general.H(M)

    # completeness
    M_sum      = np.sum(M, axis=0)
    bool_compl = np.any(np.abs(np.real(M_sum) - np.eye(len(M[0]))) < prec) and np.any(np.imag(M_sum) < prec)

    # positivity
    w        = LA.eig(M)[0]
    bool_pos = np.all(w >= -prec)

    return bool_herm and bool_compl and bool_pos
