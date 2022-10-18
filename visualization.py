import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

import general
import const


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
