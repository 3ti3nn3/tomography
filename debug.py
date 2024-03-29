import general
import mixed
import pure
import inversion
import mle
import check
import general
import state
import align
import visualization
import onestep as os
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

def ost():
    name  = 'test_ost'
    path  = ''
    new   = True
    debug = True

    d = {}
    d['dim']        = 2
    d['N_min']      = int(1e01)
    d['N_max']      = int(1e05)
    d['N_ticks']    = 20
    d['N_mean']     = 750
    d['povm_name']  = 'Pauli-4'
    d['f_sample']   = pure.sample_unitary
    d['f_estimate'] = inversion.linear
    d['f_distance'] = general.infidelity

    ost = os.OneStepTomography(name, path, new, debug, d)
    ost.parameter_report()
    ost.update_param('N_min', int(1e01))
    ost.get_originals()
    ost.get_estimates()
    ost.get_distances()
    ost.get_valids()
    ost.get_scaling()
    ost.create_originals()
    ost.reconstruct()
    ost.calculate_fitparam()
    ost.plot_distance()
    ost.plot_validity()
    ost.dispatch_model()
    ost = os.OneStepTomography(name, path, False, debug, d=None)
    ost.plot_distance()
    ost.dispatch_model(path=path)


def osc():
    name  = 'osc'
    path  = 'results/mle/onestep/'
    debug = True

    name_list = ['mle_pauli4', 'mle_pauli6', 'mle_sic']

    osc = os.OneStepComparison(name, path, debug, name_list)
    osc.parameter_report()
    osc.get_estimation_method()
    osc.get_povm_name()
    osc.get_N_min()
    osc.compare_distance(osc.transform_citeria('f_estimate'), osc.transform_citeria('povm_name'))
    osc.dispatch_model()


def tensorproduct():

    # first test
    print('first test tensorproduct')
    A = np.arange(16).reshape(4, 4)
    B = np.ones((2, 2))

    AB1 = general.tensorproduct(A, B)
    AB2 = np.kron(B, A)

    if np.all(AB1==AB2):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(AB1-AB2))}")
        print(f"componentwise difference: {AB1-AB2}")

    # second test
    print('second test tensorproduct')
    A = np.repeat(np.arange(4).reshape(1, 2, 2), 2, axis=0)
    B = np.ones((2, 2, 2))

    AB1 = general.tensorproduct(A, B)[0]
    AB2 = np.kron(B[0], A[0])

    if np.all(AB1==AB2):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(AB1-AB2))}")
        print(f"componentwise difference: {AB1-AB2}")

    # third test
    print('third test tensorproduct')
    A = np.arange(16).reshape(4, 4)
    B = np.ones((2, 2, 2))

    AB1 = general.tensorproduct(A, B)[0]
    AB2 = np.kron(B[0], A)

    if np.all(AB1==AB2):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(AB1-AB2))}")
        print(f"componentwise difference: {AB1-AB2}")

    # fourth test
    print('fourth test tensorproduct')
    A = np.repeat(np.arange(4).reshape(1, 2, 2), 2, axis=0)
    B = np.ones((4, 4))

    AB1 = general.tensorproduct(A, B)[0]
    AB2 = np.kron(B, A[0])

    if np.all(AB1==AB2):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(AB1-AB2))}")
        print(f"componentwise difference: {AB1-AB2}")


def tensorproduct_cum():

    # first test
    print('first test tensorproduct_cum')
    A = B = np.arange(16).reshape(4, 4)

    C = general.tensorproduct_cum(np.array([A, B]))
    D = general.tensorproduct(A, B)

    if np.all(C==D):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(C-D))}")
        print(f"componentwise difference: {C-D}")


    # second test
    print('second test tensorproduct_cum')
    a = b = c = np.arange(16).reshape(4, 4)

    A = general.tensorproduct_cum(np.array([a, b, c]))
    B = general.tensorproduct(general.tensorproduct(a, b), c)

    if np.all(A==B):
        print(f"tensorproduct successful!")
    else:
        print(f"tensorproduct not successful!")
        print(f"absolute difference: {np.sum(np.abs(A-B))}")
        print(f"componentwise difference: {A-B}")


def partial_trace():

    A = np.arange(16).reshape(4, 4)
    B = np.flip(np.arange(4)).reshape(2, 2)
    C = np.arange(64).reshape(8, 8)

    AB  = general.tensorproduct(A, B)
    BC  = general.tensorproduct(B, C)
    ABC = general.tensorproduct(AB, C)

    # first test
    print(f"first test partial_trace")
    AB0 = general.partial_trace(AB, [0, 1])/np.trace(A)
    if np.all(B==AB0):
        print(f"partial_trace successful!")
    else:
        print(f"partial_trace not successful!")
        print(f"absolute difference: {np.sum(np.abs(B-AB0))}")
        print(f"partial trace:\n {AB0}")

    # second test
    print(f"second test partial_trace")
    AB0 = general.partial_trace(AB, 2)/np.trace(B)
    if np.all(A==AB0):
        print(f"partial_trace successful!")
    else:
        print(f"partial_trace not successful!")
        print(f"absolute difference: {np.sum(np.abs(A-AB0))}")
        print(f"partial trace:\n {AB0}")

    # third test
    print(f"third test partial trace")
    ABC0 = general.partial_trace(ABC, [3, 4, 5])/np.trace(C)
    if np.all(AB==ABC0):
        print(f"partial_trace successful!")
    else:
        print(f"partial_trace not successful!")
        print(f"absolute difference: {np.sum(np.abs(AB-ABC0))}")
        print(f"partial trace:\n {ABC0}")

    # fourth test
    print(f"fourth test partial trace")
    ABC0 = general.partial_trace(ABC, [0, 1])/np.trace(A)
    if np.all(BC==ABC0):
        print(f"partial_trace successful!")
    else:
        print(f"partial_trace not successful!")
        print(f"absolute difference: {np.sum(np.abs(BC-ABC0))}")
        print(f"partial trace:\n {ABC0}")


def eigenbasis():

    dim = 4
    rho = pure.sample_product_unitary(dim, 1)

    MA0 = general.pauli6(2)[0]
    MB0 = general.pauli6(2)[1]

    MA1 = align.eigenbasis(general.partial_trace(rho, 0), MA0)
    MB1 = align.eigenbasis(general.partial_trace(rho, 1), MB0)

    M0 = general.tensorproduct(MA0, MB0)
    M1 = align.eigenbasis(rho, M0)

    if np.all( M1-general.tensorproduct(MA1, MB1)):
        print(f"realign successful!")
    else:
        print(f"realign not successful!")


def product_eigenbasis():

    print(f"first test product_eigenbasis")
    dim = 2
    rho = mixed.sample_hilbert(dim, 1)
    M0  = general.pauli6(dim)

    MA1 = align.product_eigenbasis(rho, M0)
    MB1 = align.eigenbasis(rho, M0)
    if np.all(MA1==MB1):
        print(f"realign successful")
    else:
        print(f"realign not successful")

    print(f"second test product_eigenbasis")
    dim = 4
    rho = mixed.sample_hilbert(dim, 1)

    M0  = general.pauli6(dim)

    _, U0 = LA.eigh(general.partial_trace(rho, 1))
    _, U1 = LA.eigh(general.partial_trace(rho, 0))
    U     = np.kron(U1, U0)

    MA1 = align.product_eigenbasis(rho, M0)
    MB1 = U@M0@general.H(U)
    if np.all(MA1==MB1):
        print(f"realign successful")
    else:
        print(f"realign not successful")
        print(f"absolute difference: {np.sum(np.abs(MA1-MB1))}")
        print(f"MA1:\n {MA1[6]}")
        print(f"MB1:\n {MB1[6]}")


def align():

    print(f"testing unit vector")
    phi   = np.array([0, 0, 0, 0, 0])*np.pi
    theta = np.array([1/2, 1/2, 1/2, 1/2, 1/2])*np.pi
    n     = np.empty((5, 3))

    b              = qt.Bloch()
    b.point_marker = 'o'
    b.point_color  = ['blue']
    b.vector_color = ['red']
    b.vector_width = 1
    b.point_size   = [10]

    for i in range(5):
        n[i] = general.n(phi[i], theta[i])
        b.add_points(n[i])

    b.render()
    plt.show()

    print(f"testing rotation")
    rho = np.zeros((2, 2))
    rho[0,0] = 1
    theta = np.array([0, 1/2])*np.pi

    for i in range(len(theta)):
        R = general.R(general.n(0, 1/2*np.pi), theta[i])
        print(R)
        visualization.qubit(vectors=R@rho@general.H(R))


def schmidt_decomp():

    print(f"testing schmidt_decomp")
    dim = 4
    prec = 1e-10


    # test state
    print(f"testing unentangled pure state")
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1

    p, U, S, V, m = general.schmidt_decomp(rho)

    r = len(p)
    n = int(np.sqrt(dim))

    assert p.shape==(r, ), f"Unexpected shape of p: {p}"
    assert U.shape==(r, n, n), f"Unexpected shape of U: {U}."
    assert S.shape==(r, n, ), f"Unexpected shape of S: {S}"
    assert V.shape==(r, n, n), f"Unexpected shape of V: {V}."
    print(f"m = {m}")

    assert np.sum(p)-1<prec, f"p {p} does not add up to 1: {np.sum(p)}"
    assert np.abs(np.sum( np.sum(np.transpose(U, axes=[0, 2, 1])@np.conjugate(U), axis=0) - r*np.eye(2)))<prec, f"U not unitary."
    assert np.all(S>=-prec), f"Schmidt coefficients should be non-negative. Instead: {S}"
    assert np.all(np.abs(np.sum(S*S, axis=1)-1)<prec), f"Square of the Schmidt coefficient should sum up to 1. Instead: {np.sum(S*S, axis=1)}"
    assert np.abs(np.sum( np.sum(np.transpose(V, axes=[0, 2, 1])@np.conjugate(V), axis=0) - r*np.eye(2)))<prec, f"V not unitary."

    rho_rec = np.zeros((dim, dim), dtype=complex)
    for i in range(r):
        psi = np.zeros(dim, dtype=complex)
        for j in range(m[i]):
            u, v = U[i,j], V[i,j]
            psi += S[i,j] * (u[:,None]@v[None,:]).flatten()

        rho_rec += p[i]* psi[:,None]@np.conjugate(psi[None,:])

    assert check.state(rho), f"rho is not a valid physical state!"

    if np.sum(np.abs(rho_rec-rho))< prec:
        print(f"testing unentangled pure state successful!")
    else:
        print(f"p = {p}")
        print(f"U = {U}")
        print(f"S = {S}")
        print(f"V = {V}")
        print(f"m = {m}")
        print(f"absolute norm = {np.sum(np.abs(rho_rec-rho))}")


    # unentangled pure states
    print(f"testing unentangled pure state")
    rho = pure.sample_product_unitary(dim, 1)

    p, U, S, V, m = general.schmidt_decomp(rho)

    r = len(p)
    n = int(np.sqrt(dim))

    assert p.shape==(r, ), f"Unexpected shape of p: {p}"
    assert U.shape==(r, n, n), f"Unexpected shape of U: {U}."
    assert S.shape==(r, n, ), f"Unexpected shape of S: {S}"
    assert V.shape==(r, n, n), f"Unexpected shape of V: {V}."
    print(f"m = {m}")

    assert np.sum(p)-1<prec, f"p {p} does not add up to 1: {np.sum(p)}"
    assert np.abs(np.sum( np.sum(np.transpose(U, axes=[0, 2, 1])@np.conjugate(U), axis=0) - r*np.eye(2)))<prec, f"U not unitary."
    assert np.all(S>=-prec), f"Schmidt coefficients should be non-negative. Instead: {S}"
    assert np.all(np.abs(np.sum(S*S, axis=1)-1)<prec), f"Square of the Schmidt coefficient should sum up to 1. Instead: {np.sum(S*S, axis=1)}"
    assert np.abs(np.sum( np.sum(np.transpose(V, axes=[0, 2, 1])@np.conjugate(V), axis=0) - r*np.eye(2)))<prec, f"V not unitary."

    rho_rec = np.zeros((dim, dim), dtype=complex)
    for i in range(r):
        psi = np.zeros(dim, dtype=complex)
        for j in range(m[i]):
            u, v = U[i,j], V[i,j]
            psi += S[i,j] * (u[:,None]@v[None,:]).flatten()

        rho_rec += p[i]* psi[:,None]@np.conjugate(psi[None,:])

    assert check.state(rho), f"rho is not a valid physical state!"

    if np.sum(np.abs(rho_rec-rho))< prec:
        print(f"testing unentangled pure state successful!")
    else:
        print(f"p = {p}")
        print(f"U = {U}")
        print(f"S = {S}")
        print(f"V = {V}")
        print(f"m = {m}")
        print(f"absolute norm = {np.sum(np.abs(rho_rec-rho))}")


    # unentangled mixed states
    print(f"testing unentangled mixed state")
    rho = mixed.sample_product_hilbert(dim, 1)

    p, U, S, V, m = general.schmidt_decomp(rho)

    r = len(p)
    n = int(np.sqrt(dim))

    assert p.shape==(r, ), f"Unexpected shape of p: {p}"
    assert U.shape==(r, n, n), f"Unexpected shape of U: {U}."
    assert S.shape==(r, n, ), f"Unexpected shape of S: {S}"
    assert V.shape==(r, n, n), f"Unexpected shape of V: {V}."
    print(f"m = {m}")

    assert np.sum(p)-1<prec, f"p {p} does not add up to 1: {np.sum(p)}"
    assert np.abs(np.sum( np.sum(np.transpose(U, axes=[0, 2, 1])@np.conjugate(U), axis=0) - r*np.eye(2)))<prec, f"U not unitary."
    assert np.all(S>=-prec), f"Schmidt coefficients should be non-negative. Instead: {S}"
    assert np.all(np.abs(np.sum(S*S, axis=1)-1)<prec), f"Square of the Schmidt coefficient should sum up to 1. Instead: {np.sum(S*S, axis=1)}"
    assert np.abs(np.sum( np.sum(np.transpose(V, axes=[0, 2, 1])@np.conjugate(V), axis=0) - r*np.eye(2)))<prec, f"V not unitary."

    rho_rec = np.zeros((dim, dim), dtype=complex)
    for i in range(r):
        psi = np.zeros(dim, dtype=complex)
        for j in range(m[i]):
            u, v = U[i,j], V[i,j]
            psi += S[i,j] * (u[:,None]@v[None,:]).flatten()

        rho_rec += p[i]* psi[:,None]@np.conjugate(psi[None,:])

    assert check.state(rho), f"rho is not a valid physical state!"

    if np.sum(np.abs(rho_rec-rho))< prec:
        print(f"testing unentangled mixed state successful!")
    else:
        print(f"p = {p}")
        print(f"U = {U}")
        print(f"S = {S}")
        print(f"V = {V}")
        print(f"m = {m}")
        print(f"absolute norm = {np.sum(np.abs(rho_rec-rho))}")


    # entangled pure states
    print(f"testing entangled pure state")
    rho = pure.sample_unitary(dim, 1)

    p, U, S, V, m = general.schmidt_decomp(rho)

    r = len(p)
    n = int(np.sqrt(dim))

    assert p.shape==(r, ), f"Unexpected shape of p: {p}"
    assert U.shape==(r, n, n), f"Unexpected shape of U: {U}."
    assert S.shape==(r, n, ), f"Unexpected shape of S: {S}"
    assert V.shape==(r, n, n), f"Unexpected shape of V: {V}."
    print(f"m = {m}")

    assert np.sum(p)-1<prec, f"p {p} does not add up to 1: {np.sum(p)}"
    assert np.abs(np.sum( np.sum(np.transpose(U, axes=[0, 2, 1])@np.conjugate(U), axis=0) - r*np.eye(2)))<prec, f"U not unitary."
    assert np.all(S>=-prec), f"Schmidt coefficients should be non-negative. Instead: {S}"
    assert np.all(np.abs(np.sum(S*S, axis=1)-1)<prec), f"Square of the Schmidt coefficient should sum up to 1. Instead: {np.sum(S*S, axis=1)}"
    assert np.abs(np.sum( np.sum(np.transpose(V, axes=[0, 2, 1])@np.conjugate(V), axis=0) - r*np.eye(2)))<prec, f"V not unitary."

    rho_rec = np.zeros((dim, dim), dtype=complex)
    for i in range(r):
        psi = np.zeros(dim, dtype=complex)
        for j in range(m[i]):
            u, v = U[i,j], V[i,j]
            psi += S[i,j] * (u[:,None]@v[None,:]).flatten()

        rho_rec += p[i]* psi[:,None]@np.conjugate(psi[None,:])

    assert check.state(rho), f"rho is not a valid physical state!"

    if np.sum(np.abs(rho_rec-rho))< prec:
        print(f"testing entangled pure state successful!")
    else:
        print(f"p = {p}")
        print(f"U = {U}")
        print(f"S = {S}")
        print(f"V = {V}")
        print(f"m = {m}")
        print(f"absolute norm = {np.sum(np.abs(rho_rec-rho))}")


    # entangled mixed states
    print(f"testing entangled mixed state")
    rho = mixed.sample_hilbert(dim, 1)

    p, U, S, V, m = general.schmidt_decomp(rho)

    r = len(p)
    n = int(np.sqrt(dim))

    assert p.shape==(r, ), f"Unexpected shape of p: {p}"
    assert U.shape==(r, n, n), f"Unexpected shape of U: {U}."
    assert S.shape==(r, n, ), f"Unexpected shape of S: {S}"
    assert V.shape==(r, n, n), f"Unexpected shape of V: {V}."
    print(f"m = {m}")

    assert np.sum(p)-1<prec, f"p {p} does not add up to 1: {np.sum(p)}"
    assert np.abs(np.sum( np.sum(np.transpose(U, axes=[0, 2, 1])@np.conjugate(U), axis=0) - r*np.eye(2)))<prec, f"U not unitary."
    assert np.all(S>=-prec), f"Schmidt coefficients should be non-negative. Instead: {S}"
    assert np.all(np.abs(np.sum(S*S, axis=1)-1)<prec), f"Square of the Schmidt coefficient should sum up to 1. Instead: {np.sum(S*S, axis=1)}"
    assert np.abs(np.sum( np.sum(np.transpose(V, axes=[0, 2, 1])@np.conjugate(V), axis=0) - r*np.eye(2)))<prec, f"V not unitary."

    rho_rec = np.zeros((dim, dim), dtype=complex)
    for i in range(r):
        psi = np.zeros(dim, dtype=complex)
        for j in range(m[i]):
            u, v = U[i,j], V[i,j]
            psi += S[i,j] * (u[:,None]@v[None,:]).flatten()

        rho_rec += p[i]* psi[:,None]@np.conjugate(psi[None,:])

    assert check.state(rho), f"rho is not a valid physical state!"

    if np.sum(np.abs(rho_rec-rho))< prec:
        print(f"testing entangled mixed state successful!")
    else:
        print(f"p = {p}")
        print(f"U = {U}")
        print(f"S = {S}")
        print(f"V = {V}")
        print(f"m = {m}")
        print(f"absolute norm = {np.sum(np.abs(rho_rec-rho))}")


if __name__ == '__main__':
    schmidt_decomp()
