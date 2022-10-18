import numpy as np


def purity(rhos: np.array, prec=1e-14):
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
        print('The follwing {0} states are not pure:\n'.format(np.sum(shadow)), rhos[shadow])
        print('Purity:\n', purity[shadow])
        return False


def povm(M: np.array):
    '''
    Checks a given set of matrices whether they are a POVM or not.

    :param M: Nxdxd dimensional array where d is the dimension of the considered system
    :return: boolean wheter the criteria of a POVM are fulfilled by the given set
    '''
    # hermitian
    MH = np.transpose(np.conjugate(M), axis1=-2, axis2=-1)
    bool_herm = M == MH

    # completeness
    prec  = 1e-15
    M_sum = np.sum(M, axis=0)
    bool_compl = np.any(np.abs(np.real(M_sum) - np.eye(len(M[0]))) < prec) and np.any(np.imag(M_sum) < prec)

    # positivity
    evalues, _ = np.linalg.eig(M)
    bool_pos   = np.all(evalues >= 0)

    return bool_herm and bool_compl and bool_pos
