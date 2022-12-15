import general
import mixed
import pure
import inversion
import mle
import general
import onestep as os
import numpy as np


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


def realign():

    mirror = False
    rho = pure.sample_product_unitary(4, 1)

    MA0 = general.pauli6(2)[0]
    MB0 = general.pauli6(2)[1]

    MA1 = general.realign_povm(general.partial_trace(rho, 0), MA0, mirror=mirror)
    MB1 = general.realign_povm(general.partial_trace(rho, 1), MB0, mirror=mirror)

    M0 = general.tensorproduct(MA0, MB0)
    M1 = general.realign_povm(rho, M0)

    if np.all( M1-general.tensorproduct(MA1, MB1)):
        print(f"realign successful!")
    else:
        print(f"realign not successful!")


if __name__ == '__main__':
    realign()
