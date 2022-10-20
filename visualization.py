import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

import general
import const
import simulate


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


def dependency_N(rho_0: np.array, N_max: int, iter=1000, M=const.spovm):
    '''
    Plots the N dependency of the Hilbert-Schmidt distance.

    :param rho_0 : state to be reconstrected
    :param N_max : maximal N
    :param iter  : number of iteration needed for the maximum likelihood method
    :param M     : array of POVMs needed for linear inversion
    :return: N-HSD plots for mle and linear inversion, developmemt of the reconstruction
        visualized on the Bloch sphere
    '''
    N_points = 10
    dim      = rho_0.shape[0]
    N        = np.ceil(np.linspace(1, N_max, N_points))

    rho_mle  = np.empty((N_points, dim, dim), dtype=np.complex)
    rho_inv  = np.empty((N_points, dim, dim), dtype=np.complex)
    hsd_mle  = np.empty(N_points, dtype=np.float)
    hsd_inv  = np.empty(N_points, dtype=np.float)

    # calculate data points for plots
    for i in range(N_points):
        axes  = np.random.randint(0, high=4, size=int(N[i]))
        D     = simulate.measure(rho_0, axes)

        rho_1      = simulate.recons(D, iter=iter)
        rho_mle[i] = rho_1
        hsd_mle[i] = general.hilbert_schmidt_distance(rho_0, rho_1)

        rho_2      = simulate.recons(D, M=M, method='inversion')
        rho_inv[i] = rho_2
        hsd_inv[i] = general.hilbert_schmidt_distance(rho_0, rho_2)

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    fig.suptitle('N-scaling of Hilber-Schmidt distance')

    axs[0].plot(N, hsd_mle, c='blue')
    axs[0].plot(N, hsd_inv, c='violet')
    axs[1].plot(N, hsd_mle, c='blue')
    axs[2].plot(N, hsd_inv, c='violet')

    axs[0].set_title('Comparison')
    axs[1].set_title('Maximum likelihood estimate')
    axs[2].set_title('Linear inversion')

    axs[0].set_xlabel(r'$N$')
    axs[1].set_xlabel(r'$N$')
    axs[2].set_xlabel(r'$N$')

    axs[0].set_xlim(1, N_max)
    axs[1].set_xlim(1, N_max)
    axs[2].set_xlim(1, N_max)

    axs[0].set_ylabel('Hilbert-Schmidt distance')

    plt.show()

    # show developmemt of the
    qubit_3(('Original', rho_0), ('MLE', rho_mle), ('Inversion', rho_inv))


def dependency_iter(rho_0: np.array, iter_max: int, N=200):
    '''
    Plots the iter dependency of the Hilbert-Schmidt distance.

    :param rho_0   : state to be reconstructed
    :param iter_max: maximal number of iterations
    :param N       : number of measurments
    :return: plots for iteration dependency of the Hilber-Schmidt distance
    '''
    iter_points = 10
    dim         = rho_0.shape[0]
    iter        = np.ceil(np.linspace(1, iter_max, iter_points))

    rho_mle  = np.empty((iter_points, dim, dim), dtype=np.complex)
    hsd_mle  = np.empty(iter_points, dtype=np.float)

    # calculate data points for plots
    for i in range(iter_points):
        axes  = np.random.randint(0, high=4, size=int(N))
        D     = simulate.measure(rho_0, axes)

        rho        = simulate.recons(D, iter=int(iter[i]))
        rho_mle[i] = rho
        hsd_mle[i] = general.hilbert_schmidt_distance(rho_0, rho)

    # plots
    plt.figure(figsize=(15, 9))

    plt.title('Iteration-scaling of Hilber-Schmidt distance')

    plt.plot(iter, hsd_mle, c='blue')
    plt.xlabel('iterations')
    plt.xlim(1, iter_max)
    plt.ylabel('Hilbert-Schmidt distance')

    plt.show()

    # show developmemt of the
    qubit_3(('Original', rho_0), ('MLE', rho_mle))
