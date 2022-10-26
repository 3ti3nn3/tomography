import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import pandas as pd

import general
import const
import simulate
import mle
import inversion
import speed

def qubit(points=np.array([None]), states=np.array([None]), kind='point', angles=[-60, 30]):
    '''
    Depending on the parameters the function creates vectors or points on the Bloch sphere.

    :param points: Nx3 array of expectations values (<sigma_x>, <sigma_y>, <sigma_z>)
    :param states: list of N Quatum objects
    :param kind  : str which should be either "point" or "vector"
    :param angles: list of angles from which the Bloch sphere is seen
    '''
    b              = qt.Bloch()
    b.point_marker = 'o'
    b.point_color  = ['blue']
    b.vector_color = ['red']
    b.vector_width = 1
    b.point_size   = [10]
    b.view         = angles

    if np.all(points != None):
        b.add_points(points)
        b.render()
    if np.all(states != None):
        b.add_states(states, kind=kind)
        b.render()
    plt.show()


def qubit_3(rho_0: (str, np.array), rho_1=(None, np.array([None])), rho_2=(None, np.array([None])), angles=[-60, 30]):
    '''
    Visualizes three different states.

    :param rho_1: first state in density representation, dataype: (name, state)
    :param rho_2: second state in density representation, dataype: (name, array of states)
    :param rho_3: third state in densitu representation, dataype: (name, array of states)
    '''
    b              = qt.Bloch()
    b.point_marker = 'o'
    b.point_color  = np.concatenate((np.repeat('red', len(rho_1[1])-1), np.repeat('violet', len(rho_2[1])-1))).tolist()
    b.vector_color = ['blue', 'red', 'violet']
    b.vector_width = 2
    b.point_size   = [10]
    b.view         = angles

    vec_0 = general.expect_xyz(rho_0[1])
    b.add_vectors(vec_0)
    b.add_annotation(vec_0/2, rho_0[0], c='blue', fontsize=10)

    if rho_1[0]!=None:
        for i in range(len(rho_1[1])-1):
            rho = rho_1[1][i]
            pnt = general.expect_xyz(rho)
            b.add_points(pnt)

        rho = rho_1[1][-1]
        vec = general.expect_xyz(rho)
        b.add_vectors(vec)
        b.add_annotation(vec/2, rho_1[0], c='red', fontsize=10)
        b.render()

    if rho_2[0]!=None:
        for i in range(len(rho_2[1])-1):
            rho = rho_2[1][i]
            pnt = general.expect_xyz(rho)
            b.add_points(pnt)

        rho = rho_2[1][-1]
        vec = general.expect_xyz(rho)
        b.add_vectors(vec)
        b.add_annotation(vec/2, rho_2[0], c='violet', fontsize=10)
        b.render()

    plt.show()


def expectation_distribution(rho, n_bins=10):
    '''
    Visualizes the three different Pauli expectation values.

    :param rho: sample of state
    :param n_bins: bin width of histogramm
    ;return:
    '''
    N = len(rho)

    expect_x = general.expect(const.sx, rho)
    expect_y = general.expect(const.sy, rho)
    expect_z = general.expect(const.sz, rho)

    mu_x = np.sum(expect_x)/N
    mu_y = np.sum(expect_y)/N
    mu_z = np.sum(expect_z)/N

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(30, 6), sharey=True)
    axs[0].set_ylabel('counts')

    fig.suptitle('Expectation Values of Approach two')

    n_x, _, _ = axs[0].hist(expect_x, bins=n_bins)
    n_y, _, _ = axs[1].hist(expect_y, bins=n_bins)
    n_z, _, _ = axs[2].hist(expect_z, bins=n_bins)

    axs[0].vlines(mu_x, 0, np.max(n_x), color='red')
    axs[1].vlines(mu_y, 0, np.max(n_y), color='red')
    axs[2].vlines(mu_z, 0, np.max(n_z), color='red')

    axs[0].text(0.075, np.max(n_x)/2, '{0:.2e}'.format(mu_x), color='red')
    axs[1].text(0.075, np.max(n_x)/2, '{0:.2e}'.format(mu_y), color='red')
    axs[2].text(0.075, np.max(n_x)/2, '{0:.2e}'.format(mu_z), color='red')

    axs[0].set_title(r'Expectation value $<\sigma_x>$')
    axs[1].set_title(r'Expectation value $<\sigma_y>$')
    axs[2].set_title(r'Expectation value $<\sigma_z>$')

    axs[0].set_xlabel(r'$<\sigma_x>$')
    axs[1].set_xlabel(r'$<\sigma_y>$')
    axs[2].set_xlabel(r'$<\sigma_z>$')

    axs[0].set_xlim(-1, 1)
    axs[1].set_xlim(-1, 1)
    axs[2].set_xlim(-1, 1)

    plt.show()


def hilbert_dist(rho_0: np.array, M: np.array, N_max: int, iter=100):
    '''
    Plots the N dependency of the Hilbert-Schmidt distance.

    :param rho_0 : state to be reconstrected
    :param N_max : maximal N
    :param M     : set of POVMs
    :param iter  : number of iteration needed for the maximum likelihood method
    :return: N-HSD plots for mle and linear inversion and developememt of the reconstruction
        visualized on the Bloch sphere
    '''
    dim      = rho_0.shape[0]
    steps    = np.logspace(0, np.log(N_max), 50, dtype=np.int64)
    n_steps  = len(steps)

    rho_mle  = np.empty((n_steps, dim, dim), dtype=np.complex)
    rho_inv  = np.empty((n_steps, dim, dim), dtype=np.complex)
    hsd_mle  = np.empty(n_steps, dtype=np.float)
    hsd_inv  = np.empty(n_steps, dtype=np.float)

    D = simulate.measure(rho_0, N_max, M)

    # calculate data points for plots
    for i in range(n_steps):
        rho_1      = mle.iterative(D[:steps[i]], M, iter=iter)
        rho_mle[i] = rho_1
        hsd_mle[i] = general.hilbert_dist(rho_0, rho_1)

        rho_2      = inversion.linear(D[:steps[i]], M)
        rho_inv[i] = rho_2
        hsd_inv[i] = general.hilbert_dist(rho_0, rho_2)

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of Hilber-Schmidt distance')

    axs[0].plot(steps, hsd_mle, c='blue')
    axs[0].plot(steps, hsd_inv, c='violet')
    axs[1].plot(steps, hsd_mle, c='blue')
    axs[2].plot(steps, hsd_inv, c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title('Maximum likelihood estimate')
    axs[2].set_title('Linear inversion')

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xlim(1, N_max)
    axs[1].set_xlim(1, N_max)
    axs[2].set_xlim(1, N_max)

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    axs[0].set_ylabel('Hilbert-Schmidt distance')

    plt.show()

    # show developmemt of the
    qubit_3(('Original', rho_0), ('MLE', rho_mle), ('Inversion', rho_inv))


def hilbert_dist_iter(rho_0: np.array, M:np.array, iter_max: int, N=1000):
    '''
    Plots the iter dependency of the Hilbert-Schmidt distance.

    :param rho_0   : state to be reconstructed
    :param M       : set of POVMs
    :param iter_max: maximal number of iterations
    :param N       : number of measurments
    :return: plots for iteration dependency of the Hilber-Schmidt distance
    '''
    dim     = rho_0.shape[0]
    steps   = np.linspace(1, iter_max, 50, dtype=np.int64)
    n_steps = len(steps)

    rho_mle  = np.empty((n_steps, dim, dim), dtype=np.complex)
    hsd_mle  = np.empty(n_steps, dtype=np.float)

    D = simulate.measure(rho_0, N, M)

    # calculate data points for plots
    for i in range(n_steps):
        rho        = mle.iterative(D, M, iter=steps[i])
        rho_mle[i] = rho
        hsd_mle[i] = general.hilbert_dist(rho_0, rho)

    # plots
    plt.figure(figsize=(15, 9))

    plt.title('Iteration-scaling of Hilber-Schmidt distance')

    plt.plot(steps, hsd_mle, c='blue')
    plt.xlabel('iterations')
    plt.xlim(1, iter_max)
    plt.ylabel('Hilbert-Schmidt distance')
    plt.yscale('log')

    plt.show()

    # show developmemt of the
    qubit_3(('Original', rho_0), ('MLE', rho_mle))


def bures_dist(rho_0: np.array, M: np.array, N_max: int, iter=50):
    '''
    Plots the N dependency of Bures distance.

    :param rho_0 : state to be reconstrected
    :param M     : set of POVMs
    :param N_max : maximal N
    :param iter  : number of iteration needed for the maximum likelihood method
    :return: N-bures_dist plots for mle and linear inversion
    '''
    dim      = rho_0.shape[0]
    steps    = np.logspace(0, np.log(N_max), 20, dtype=np.int64)
    n_steps  = len(steps)

    rho_mle  = np.empty((n_steps, dim, dim), dtype=np.complex)
    rho_inv  = np.empty((n_steps, dim, dim), dtype=np.complex)
    bur_mle  = np.empty(n_steps, dtype=np.float)
    bur_inv  = np.empty(n_steps, dtype=np.float)

    D = simulate.measure(rho_0, N_max, M)

    # calculate data points for plots
    for i in range(n_steps):
        print(i)

        rho_1      = mle.iterative(D[:steps[i]], M, iter=iter)
        rho_mle[i] = rho_1
        bur_mle[i] = general.bures_dist(rho_0, rho_1)
        print('Maximum Likelihood estimate done!')

        rho_2      = inversion.linear(D[:steps[i]], M)
        rho_inv[i] = rho_2
        try:
            bur_inv[i] = general.bures_dist(rho_0, rho_2)
        except:
            bur_inv[i] = 1-general.fidelity(rho_0, rho_2)
        print('Linear Inversion done!')

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of Bures distance')

    axs[0].plot(steps, bur_mle, c='blue')
    axs[0].plot(steps, bur_inv, c='violet')
    axs[1].plot(steps, bur_mle, c='blue')
    axs[2].plot(steps, bur_inv, c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title('Maximum likelihood estimate')
    axs[2].set_title('Linear inversion')

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xlim(1, N_max)
    axs[1].set_xlim(1, N_max)
    axs[2].set_xlim(1, N_max)

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    axs[0].set_ylabel('Bures distance')

    plt.show()

    # show developmemt of the
    qubit_3(('Original', rho_0), ('MLE', rho_mle))


def infidelity(rho_0: np.array, func1, func2, M1: np.array, M2: np.array, N_max: int):
    '''
    Plots the N dependency of infidelity.

    :param rho_0 : state to be reconstrected
    :param func1 : first function how to create a estimate
        datatype: tuple (describtion: str, function)
    :param func2 : second function how to create a estimate
        datatype: tuple (describtion: str, function)
    :param M1    : set of POVMs for func1
    :param M2    : set of POVMs for func2
    :param N_max : maximal N
    :return: N-bures_dist plots for mle and linear inversion
    '''
    dim      = rho_0.shape[0]
    steps    = np.logspace(0, np.log(N_max), 20, dtype=np.int64)
    n_steps  = len(steps)

    rho_m1  = np.zeros((n_steps, dim, dim), dtype=np.complex)
    rho_m2  = np.zeros((n_steps, dim, dim), dtype=np.complex)
    inf_m1  = np.zeros(n_steps, dtype=np.float)
    inf_m2  = np.zeros(n_steps, dtype=np.float)

    if np.all(M1==M2):
        D  = simulate.measure(rho_0, N_max, M1)
        D1 = D
        D2 = D
    else:
        D1 = simulate.measure(rho_0, N_max, M1)
        D2 = simulate.measure(rho_0, N_max, M2)

    # calculate data points for plots
    for i in range(n_steps):
        rho_m1[i] = func1[1](D1[:steps[i]], M1)
        inf_m1[i] = 1-general.fidelity(rho_0, rho_m1[i])

        rho_m2[i] = func2[1](D2[:steps[i]], M2)
        inf_m2[i] = 1-general.fidelity(rho_0, rho_m2[i])

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of infidelity')

    axs[0].plot(steps, inf_m1, c='blue', label=func1[0])
    axs[0].plot(steps, inf_m2, c='violet', label=func2[0])
    axs[1].plot(steps, inf_m1, c='blue')
    axs[2].plot(steps, inf_m2, c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title(func1[0])
    axs[2].set_title(func2[0])

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xlim(1, N_max)
    axs[1].set_xlim(1, N_max)
    axs[2].set_xlim(1, N_max)

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    if func1[1]!=inversion.linear and func2[1]!=inversion.linear:
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
    elif func1[1]==inversion.linear:
        axs[2].set_yscale('log')
    elif func2[1]==inversion.linear:
        axs[1].set_yscale('log')

    axs[0].set_ylabel('infidelity')
    axs[0].legend()

    plt.show()


def speed_comparison(title, iterations=10, **kwargs):
    '''
    Shows the result of speed comparison of arbitrary functions.

    :param title     : title of the plot
    :param iterations: number of iterations the test function is tested
    :param **kwargs  : dictionary like objekt of the form "name = (func, list of parameters)"
    :return: dictionaries of times each test function needed
    '''
    data = speed.compare(iterations=10, **kwargs)
    df   = pd.DataFrame.from_dict(data, orient='index')

    ax = df.plot.bar(figsize=(10, 6), ylabel='time', title=title , legend=False, rot=0)
    ax.plot()

    plt.show()
