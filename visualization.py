import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import pandas as pd
from scipy.optimize import curve_fit

import general
import const
import simulate
import mle
import inversion
import speed
import pure
import mixed

def qubit(points=np.array([None]), states=np.array([None]), kind='point', angles=[-60, 30]):
    '''
    Depending on the parameters the function creates vectors or points on the Bloch sphere.

    :param points: Nx2x2 of states in density representation
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
        try:
            b.add_points(general.expect_xyz(points).T)
        except:
            b.add_points(general.expect_xyz(points))
        b.render()
    if np.all(states != None):
        b.add_states(states, kind=kind)
        b.render()
    plt.show()


def qubit_3(rho_0: (str, np.array), rho_1=(None, np.array([None])), rho_2=(None, np.array([None])), angles=[-60, 30]):
    '''
    Visualizes three different states.

    :param rho_1 : first state in density representation, dataype: (name, state)
    :param rho_2 : second state in density representation, dataype: (name, array of states)
    :param rho_3 : third state in density representation, dataype: (name, array of states)
    :param angles: angles of the perspective of the Bloch sphere
    :return:
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
    Visualizes the Pauli expectation values.

    :param rho   : sample of state
    :param n_bins: bin width of histogramm
    :return:
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


def hilbert_dist(rho_0: np.array, M: np.array, N_max: int, n_mean: int, iter=100, qubit=False):
    '''
    Plots the N dependency of the Hilbert-Schmidt distance.

    :param rho_0 : state to be reconstrected
    :param N_max : maximal N
    :param M     : set of POVMs
    :param iter  : number of iteration needed for the maximum likelihood method
    :return:
    '''
    dim      = rho_0.shape[0]
    steps    = np.logspace(2, np.log10(N_max), 10, dtype=np.int64)
    N_steps  = len(steps)

    rho_m1  = np.zeros((D_mean, N_steps, dim, dim), dtype=np.complex)
    rho_m2  = np.zeros((D_mean, N_steps, dim, dim), dtype=np.complex)
    hsd_m1  = np.zeros((D_mean, N_steps), dtype=np.float)
    hsd_m2  = np.zeros((D_mean, N_steps), dtype=np.float)

    for j in range(D_mean):
        if np.all(M1==M2):
            D  = simulate.measure(rho_0, N_max, M1)
            D1 = D
            D2 = D
        else:
            D1 = simulate.measure(rho_0, N_max, M1)
            D2 = simulate.measure(rho_0, N_max, M2)

        # calculate data points for plots
        for i in range(N_steps):
            rho_m1[j,i] = func1[1](D1[:steps[i]], M1)
            hsd_m1[j,i] = 1-general.fidelity(rho_0, rho_m1[j,i])

            rho_m2[j,i] = func2[1](D2[:steps[i]], M2)
            hsd_m2[j,i] = 1-general.fidelity(rho_0, rho_m2[j,i])

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of Hilber-Schmidt distance')

    axs[0].plot(steps, np.mean(hsd_mle, axis=0), c='blue')
    axs[0].plot(steps, np.mean(hsd_inv, axis=0), c='violet')
    axs[1].plot(steps, np.mean(hsd_mle, axis=0), c='blue')
    axs[2].plot(steps, np.mean(hsd_inv, axis=0), c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title('Maximum likelihood estimate')
    axs[2].set_title('Linear inversion')

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    if np.all(inf_m1>0) and np.all(inf_m2>0):
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
    elif np.all(inf_m1>0):
        axs[1].set_yscale('log')
    elif np.all(inf_m2>0):
        axs[2].set_yscale('log')

    axs[0].set_ylabel('Hilbert-Schmidt distance')

    plt.show()

    # show developmemt of the
    if qubit:
        qubit_3(('Original', rho_0), ('MLE', rho_mle), ('Inversion', rho_inv))


def hilbert_dist_iter(rho_0: np.array, M:np.array, iter_max: int, D_mean: int, N=1000, qubit=False):
    '''
    Plots the iter dependency of the Hilbert-Schmidt distance.

    :param rho_0   : state to be reconstructed
    :param M       : set of POVMs
    :param iter_max: maximal number of iterations
    :param N       : number of measurments
    :return:
    '''
    dim     = rho_0.shape[0]
    steps   = np.linspace(1, iter_max, 50, dtype=np.int64)
    N_steps = len(steps)

    rho_mle  = np.empty((D_mean, N_steps, dim, dim), dtype=np.complex)
    hsd_mle  = np.empty((D_mean, N_steps), dtype=np.float)

    # calculate data points for plots
    for j in range(D_mean):
        D  = simulate.measure(rho_0, N_max, M1)

        for i in range(N_steps):
            rho_mle[j,i] = mle.iterative(D, M, iter=steps[i])
            hsd_mle[j,i] = general.hilbert_dist(rho_0, rho_mle[j,i])

    # plots
    plt.figure(figsize=(15, 9))

    plt.title('Iteration-scaling of Hilber-Schmidt distance')

    plt.plot(steps, np.mean(hsd_mle, axis=0), c='blue')
    plt.xlim(1, iter_max)
    plt.xlabel('iterations')
    plt.ylabel('Hilbert-Schmidt distance')
    plt.yscale('log')

    plt.show()

    # show developmemt of the
    if qubit:
        qubit_3(('Original', rho_0), ('MLE', rho_mle))


def infidelity(rho_0: np.array, func1: tuple, func2: tuple, M1: np.array, M2: np.array, N_max: int, D_mean: int, qubit=False):
    '''
    Plots the N dependency of infidelity.

    :param rho_0 : state to be reconstrected
    :param func1 : first function how to create a estimate
        datatype: tuple (description: str, function)
    :param func2 : second function how to create a estimate
        datatype: tuple (description: str, function)
    :param M1    : set of POVMs for func1
    :param M2    : set of POVMs for func2
    :param N_max : maximal size of measurement sample
    :param D_mean: number of mean measurements for mean value
    :param q     : boolean about whether the development should be plotted
    :return:
    '''
    dim      = rho_0.shape[0]
    steps    = np.logspace(2, np.log10(N_max), 10, dtype=np.int64)
    N_steps  = len(steps)

    rho_m1  = np.zeros((D_mean, N_steps, dim, dim), dtype=np.complex)
    rho_m2  = np.zeros((D_mean, N_steps, dim, dim), dtype=np.complex)
    inf_m1  = np.zeros((D_mean, N_steps), dtype=np.float)
    inf_m2  = np.zeros((D_mean, N_steps), dtype=np.float)

    for j in range(D_mean):
        if np.all(M1==M2):
            D  = simulate.measure(rho_0, N_max, M1)
            D1 = D
            D2 = D
        else:
            D1 = simulate.measure(rho_0, N_max, M1)
            D2 = simulate.measure(rho_0, N_max, M2)

        # calculate data points for plots
        for i in range(N_steps):
            rho_m1[j,i] = func1[1](D1[:steps[i]], M1)
            inf_m1[j,i] = 1-general.fidelity(rho_0, rho_m1[j,i])

            rho_m2[j,i] = func2[1](D2[:steps[i]], M2)
            inf_m2[j,i] = 1-general.fidelity(rho_0, rho_m2[j,i])

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of infidelity')

    axs[0].plot(steps, np.mean(inf_m1, axis=0), c='blue', label=func1[0])
    axs[0].plot(steps, np.mean(inf_m2, axis=0), c='violet', label=func2[0])
    axs[1].plot(steps, np.mean(inf_m1, axis=0), c='blue')
    axs[2].plot(steps, np.mean(inf_m2, axis=0), c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title(func1[0])
    axs[2].set_title(func2[0])

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    if np.all(inf_m1>0) and np.all(inf_m2>0):
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
    elif np.all(inf_m1>0):
        axs[1].set_yscale('log')
    elif np.all(inf_m2>0):
        axs[2].set_yscale('log')

    axs[0].set_ylabel('infidelity')
    axs[0].legend()

    plt.show()

    # show developmemt of the
    if qubit:
        qubit_3(('Original', rho_0), ('Method 2', rho_m1), ('Method 2', rho_m2))


def infidelity_mean(func1: tuple, func2: tuple, M1: np.array, M2: np.array, N_max: int, N_mean: int):
    '''
    Calculates and plots the infidelity averaged over N_mean different rho_0.

    :param func1 : first function how to create estimate
        datatype: tuple (description: str, function)
    :param func2 : second function how to create estimate
        datatype: tuple (description: str, function)
    :param M1    : set of POVMs for func1
    :param M2    : set of POVMs for func2
    :param N_max : maximum size of measurement sample
    :param N_mean: number of different samples
    :return:
    '''
    dim     = 2
    steps   = np.logspace(0, np.log10(N_max), 50, dtype=np.int64)
    N_steps = len(steps)
    rho_0   = pure.unitary_to_density(dim, N_mean)

    # initialize data storage
    rho_m1 = np.empty((N_mean, N_steps, dim, dim), dtype=np.complex)
    rho_m2 = np.empty((N_mean, N_steps, dim, dim), dtype=np.complex)
    inf_m1 = np.empty((N_mean, N_steps), dtype=np.float)
    inf_m2 = np.empty((N_mean, N_steps), dtype=np.float)

    # creating data
    for n in range(N_mean):
        if np.all(M1==M2):
            D  = simulate.measure(rho_0[n], N_max, M1)
            D1 = D
            D2 = D
        else:
            D1 = simulate.measure(rho_0[n], N_max, M1)
            D2 = simulate.measure(rho_0[n], N_max, M2)

        # calculate data points for plots
        for i in range(N_steps):
            rho_m1[n,i] = func1[1](D1[:steps[i]], M1)
            inf_m1[n,i] = 1-general.fidelity(rho_0[n], rho_m1[n,i])

            rho_m2[n,i] = func2[1](D2[:steps[i]], M2)
            inf_m2[n,i] = 1-general.fidelity(rho_0[n], rho_m2[n,i])

    inf_m1_mean = np.mean(inf_m1, axis=0)
    inf_m2_mean = np.mean(inf_m2, axis=0)
    inf_m1_std  = np.std(inf_m1, axis=0)
    inf_m2_std  = np.std(inf_m2, axis=0)

    # fit data
    f = lambda x, a, A: A*x**a

    threshold = 1e02
    popt_m1, pcov_m1 = curve_fit(f, steps[steps>threshold], inf_m1_mean[steps>threshold], p0=[-0.65, 0.3], sigma=inf_m1_std[steps>threshold], method='trf')
    popt_m2, pcov_m2 = curve_fit(f, steps[steps>threshold], inf_m2_mean[steps>threshold], p0=[-0.58, 0.2], sigma=inf_m2_std[steps>threshold], method='trf')

    a0_m1, A0_m1 = popt_m1-np.sqrt(np.diag(pcov_m1))
    a1_m1, A1_m1 = popt_m1+np.sqrt(np.diag(pcov_m1))

    a0_m2, A0_m2 = popt_m2-np.sqrt(np.diag(pcov_m2))
    a1_m2, A1_m2 = popt_m2+np.sqrt(np.diag(pcov_m2))

    # plot
    cont = np.logspace(0, np.log10(N_max), 100)

    plt.figure(figsize=(15, 9))

    plt.fill_between(cont, y1=f(cont, a0_m1, A0_m1), y2=f(cont, a1_m1, A1_m1), color='lightblue', alpha=0.3, label=r'$1\sigma$ environment of '+func1[0])
    plt.fill_between(cont, y1=f(cont, a0_m2, A0_m2), y2=f(cont, a1_m2, A1_m2), color='lightgreen', alpha=0.3, label=r'$1\sigma$ environment of '+func2[0])

    plt.plot(steps, inf_m1_mean, color='navy', linestyle='None', markersize=5, marker='o', label=func1[0])
    plt.plot(steps, inf_m2_mean, color='forestgreen', linestyle='None', markersize=5, marker='o', label=func2[0])

    plt.plot(cont, f(cont, *popt_m1), color='lightblue', label='Fit '+func1[0]+r', a = {0:.2e} $\pm$ {1:.2e}'.format(popt_m1[0], np.sqrt(pcov_m1[0, 0])))
    plt.plot(cont, f(cont, *popt_m2), color='lightgreen', label='Fit '+func2[0]+r', a = {0:.2e} $\pm$ {1:.2e}'.format(popt_m2[0], np.sqrt(pcov_m2[0, 0])))

    plt.title('N-scaling of mean infidelity averaged over {0} randomly picked pure states'.format(N_mean))
    plt.xlabel(r'$N$')
    plt.ylabel('mean infidelity')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, N_max)

    plt.show()


def euclidean_mean(func1: tuple, func2: tuple, M1: np.array, M2: np.array, N_max: int, N_mean: int):
    '''
    Calculates and plots the mean euclidean distance between the sample state and the estimator averaged
    over N_mean randomly picked pure states.

    :param func1 : first function how to create estimate
        datatype: tuple (description: str, function)
    :param func2 : second function how to create estimate
        datatype: tuple (description: str, function)
    :param M1    : set of POVMs for func1
    :param M2    : set of POVMs for func2
    :param N_max : maximum size of measurement sample
    :param N_mean: number of different samples
    :return:
    '''
    dim     = 2
    steps   = np.logspace(0, np.log10(N_max), 100, dtype=np.int64)
    N_steps = len(steps)
    rho_0   = pure.unitary_to_density(dim, N_mean)

    # initialize data storage
    rho_m1 = np.empty((N_mean, N_steps, dim, dim), dtype=np.complex)
    rho_m2 = np.empty((N_mean, N_steps, dim, dim), dtype=np.complex)
    euc_m1 = np.empty((N_mean, N_steps), dtype=np.float)
    euc_m2 = np.empty((N_mean, N_steps), dtype=np.float)

    # creating data
    for n in range(N_mean):
        if np.all(M1==M2):
            D  = simulate.measure(rho_0[n], N_max, M1)
            D1 = D
            D2 = D
        else:
            D1 = simulate.measure(rho_0[n], N_max, M1)
            D2 = simulate.measure(rho_0[n], N_max, M2)

        # calculate data points for plots
        for i in range(N_steps):
            rho_m1[n,i] = func1[1](D1[:steps[i]], M1)
            euc_m1[n,i] = general.euclidean_dist(rho_0[n], rho_m1[n,i])

            rho_m2[n,i] = func2[1](D2[:steps[i]], M2)
            euc_m2[n,i] = general.euclidean_dist(rho_0[n], rho_m2[n,i])

    euc_m1_mean = np.mean(euc_m1, axis=0)
    euc_m2_mean = np.mean(euc_m2, axis=0)
    euc_m1_std  = np.std(euc_m1, axis=0)
    euc_m2_std  = np.std(euc_m2, axis=0)

    # fit data
    f = lambda x, a, A: A*x**a

    threshold = 1e02
    popt_m1, pcov_m1 = curve_fit(f, steps[steps>threshold], euc_m1_mean[steps>threshold], p0=[-0.65, 0.3], sigma=euc_m1_std[steps>threshold], method='trf')
    popt_m2, pcov_m2 = curve_fit(f, steps[steps>threshold], euc_m2_mean[steps>threshold], p0=[-0.58, 0.2], sigma=euc_m2_std[steps>threshold], method='trf')

    a0_m1, A0_m1 = popt_m1-np.sqrt(np.diag(pcov_m1))
    a1_m1, A1_m1 = popt_m1+np.sqrt(np.diag(pcov_m1))

    a0_m2, A0_m2 = popt_m2-np.sqrt(np.diag(pcov_m2))
    a1_m2, A1_m2 = popt_m2+np.sqrt(np.diag(pcov_m2))

    # plot
    cont = np.logspace(0, np.log10(N_max), 100)

    plt.figure(figsize=(15, 9))

    plt.fill_between(cont, y1=f(cont, a0_m1, A0_m1), y2=f(cont, a1_m1, A1_m1), color='lightblue', alpha=0.3, label=r'$1\sigma$ environment of '+func1[0])
    plt.fill_between(cont, y1=f(cont, a0_m2, A0_m2), y2=f(cont, a1_m2, A1_m2), color='lightgreen', alpha=0.3, label=r'$1\sigma$ environment of '+func2[0])

    plt.plot(steps, euc_m1_mean, color='navy', linestyle='None', markersize=3, marker='o', label=func1[0])
    plt.plot(steps, euc_m2_mean, color='forestgreen', linestyle='None', markersize=3, marker='o', label=func2[0])

    plt.plot(cont, f(cont, *popt_m1), color='lightblue', label='Fit '+func1[0]+r', a = {0:.2e} $\pm$ {1:.2e}'.format(popt_m1[0], np.sqrt(pcov_m1[0, 0])))
    plt.plot(cont, f(cont, *popt_m2), color='lightgreen', label='Fit '+func2[0]+r', a = {0:.2e} $\pm$ {1:.2e}'.format(popt_m2[0], np.sqrt(pcov_m2[0, 0])))

    plt.title('N-scaling of mean euclidean distance averaged over {0} randomly picked pure states'.format(N_mean))
    plt.xlabel(r'$N$')
    plt.ylabel('mean euclidean distance')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, N_max)

    plt.show()


def speed_comparison(title, iterations=10, **kwargs):
    '''
    Shows the result of speed comparison of arbitrary functions.

    :param title     : title of the plot
    :param iterations: number of iterations the test function is tested
    :param **kwargs  : dictionary like objekt of the form "name = (func, list of parameters)"
    :return: 
    '''
    data = speed.compare(iterations=10, **kwargs)
    df   = pd.DataFrame.from_dict(data, orient='index')

    ax = df.plot.bar(figsize=(10, 6), ylabel='time', title=title , legend=False, rot=0)
    ax.plot()

    plt.show()
