import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import qutip as qt
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors

import general
import const
import simulate
import mle
import inversion
import speed
import pure
import mixed
import check

d = {}
d[pure.unitary_to_density]    = 'pure'
d[mixed.sample_bures]         = 'mixed'
d[mixed.sample_hilbert]       = 'mixed'
d[mixed.hermitian_to_density] = 'mixed'
d[mle.iterative]      = 'MLE'
d[mle.two_step]       = 'two step MLE'
d[inversion.linear]   = 'LI'
d[inversion.two_step] = 'two step LI'
d[general.euclidean_dist] = 'Euclidean distance'
d[general.hilbert_dist]   = 'Hilbert Schmidt distance'
d[general.infidelity]     = 'infidelity'


def qubit(points=np.array([None]), vectors=np.array([None]), states=np.array([None]), kind='point', angles=[-60, 30]):
    '''
    Depending on the parameters the function creates vectors or points on the Bloch sphere.

    :param points : Nx2x2 of states in density representation
    :param vectors: Nx2x2 of states in density representation
    :param states : list of N Quatum objects
    :param kind   : str which should be either "point" or "vector"
    :param angles : list of angles from which the Bloch sphere is seen
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
    if np.all(vectors!= None):
        try:
            b.add_vectors(general.expect_xyz(vectors))
        except:
            b.add_vectors(general.expect_xyz(vectors))
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


def eigenvalue_distribution(f_sample, N):
    '''
    Visualizes the eigenvalue distribution of a given array of states.

    :param rho   : array of states in density represnetation
    :param n_bins: number of bins
    :return:
    '''
    measure = {}
    measure[mixed.sample_hilbert] = 'Hilbert-Schmidt measure'
    measure[mixed.sample_bures]   = 'Bures measure'

    distribution = {}
    distribution[mixed.sample_hilbert] = lambda x: 12*(x-1/2)**2
    distribution[mixed.sample_bures]   = lambda x: 8*(x-1/2)**2/(np.pi*np.sqrt(x*(1-x)))

    m = measure[f_sample]
    P = distribution[f_sample]

    eig, _ = LA.eig(f_sample(2, N))
    eig    = np.real(eig.flatten())

    # plot simulation data
    plt.figure(figsize=(12, 9))
    plt.title(f'Eigenvalue distribution for sampling according to {m}')
    plt.hist(eig, int(np.sqrt(N)), color='navy', density=True)

    # plot theoretical distribution
    x = np.linspace(0, 1, 100)[1:-1]
    plt.plot(x, P(x), label='expected distribution', color='violet')

    plt.xlabel('eigenvalue')
    plt.ylabel(r'$P$')
    plt.xlim(0, 1)
    plt.legend()

    plt.show()


def plot_distance(self):
    '''
    Plots the N-dependency of the distance measure for the given simulation.
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f'N-scaling of {d[self.f_distance]} averaged over {self.N_mean} {d[self.f_sample]} states')

    # calculate mean
    mean = np.mean(self.get_distances(), axis=0, where=self.get_valids())
    std  = np.std(self.get_distances(), axis=0, where=self.get_valids())

    plt.plot(self.x_N, mean, color='navy', linestyle='None', markersize=5, marker='o', label=f'{d[self.f_estimate]} with {self.povm_name}')

    # fit curve
    try:
        f = lambda x, a, A: A*x**a

        popt, pcov = curve_fit(f, self.x_N, mean, p0=[-0.5, 0.2], sigma=std)

        param1 = popt-np.sqrt(np.diag(pcov))
        param2 = popt+np.sqrt(np.diag(pcov))

        x = np.logspace(np.log10(self.N[0]), np.log10(self.N[1]), 100, dtype=np.int32)
        plt.fill_between(x, y1=f(x, *param1), y2=f(x, *param2), color='lightblue', alpha=0.3)

        plt.plot(x, f(x, *popt), color='lightblue', label=f'corresponding fit and $1\sigma$ with a = {popt[0]:.2e} $\pm$ {pcov[0,0]:.2e}')

        self.logger.info('curve_fit successful!')
        self.logger.info(f'\n'
            'Fit parameters\n'\
            '--------------\n'\
            f'a = {popt[0]:.2e} +/- {np.sqrt(pcov[0,0]):.2e}\n'\
            f'A = {popt[1]:.2e} +/- {np.sqrt(pcov[1,1]):.2e}')
    except:
         self.logger.info('curve_fit not successful!')

    plt.xlabel(r'$N$')
    plt.ylabel(f'{d[self.f_distance]}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(self.x_N[0], self.x_N[-1])

    plt.legend()
    plt.savefig('plots/dist_'+self.name+'.png', format='png', dpi=300)


def compare_distance(self, criteria_1, criteria_2):
    '''
    Compares two Tomography schemes in one plot.

    :param criteria_1: first criteria need to be consideread, same order as self.tomo_list
    :param criteria_2: second criteria need to be considered, same order as self.tomt_list
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f'N-scaling of {d[self.f_distance]} averaged over {self.N_mean} {d[self.f_sample]} states')

    c = [['navy', 'lightblue'], ['forestgreen', 'lightgreen'], ['red', 'lightsalmon'], ['black', 'grey'], ['peru', 'sandybrown'], ['darkorange', 'bisque']]
    for idx, tomo in enumerate(self.tomo_list):

        # calculate mean
        mean = np.mean(tomo.get_distances(), axis=0, where=tomo.get_valids())
        std  = np.std(tomo.get_distances(), axis=0, where=tomo.get_valids())

        plt.plot(tomo.x_N, mean, color=c[idx][0], linestyle='None', markersize=5, marker='o', label=f'{d[tomo.f_estimate]} with {criteria_1[idx]} and {criteria_2[idx]}')

        # fit curve
        try:
            f = lambda x, a, A: A*x**a

            popt, pcov = curve_fit(f, tomo.x_N, mean, p0=[-0.5, 0.2], sigma=std)

            param1 = popt-np.sqrt(np.diag(pcov))
            param2 = popt+np.sqrt(np.diag(pcov))

            x = np.logspace(np.log10(tomo.N[0]), np.log10(tomo.N[1]), 100, dtype=np.int32)
            # plt.fill_between(x, y1=f(x, *param1), y2=f(x, *param2), color=c[idx][1], alpha=0.3)

            plt.plot(x, f(x, *popt), color=c[idx][1], label=f'fit with a = {popt[0]:.2f} $\pm$ {np.sqrt(pcov[0,0]):.2f}, A = {popt[1]:.2f} $\pm$ {np.sqrt(pcov[1,1]):.2f}')

            self.logger.info(f'curve_fit of {tomo.name} successful!')
            self.logger.info(f'\n'
                f'Fit parameters of {tomo.name}\n'\
                '--------------\n'\
                f'a = {popt[0]:.2f} +/- {np.sqrt(pcov[0,0]):.2f}\n'\
                f'A = {popt[1]:.2f} +/- {np.sqrt(pcov[1,1]):.2f}')
        except:
             self.logger.info(f'curve_fit of {tomo.name} not successful!')

    plt.xlabel(r'$N$')
    plt.ylabel(f'{d[self.f_distance]}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((self.N[0], self.N[1]))
    plt.legend()

    plt.savefig('plots/comp_'+self.name+'.png', format='png', dpi=300)


def plot_alpha_dependency(self):
    '''
    Plots the alpha dependency of the scaling.
    '''
    fig, ax1 = plt.subplots(figsize=(12,9))
    fig.suptitle(r'$\alpha$-dependency of fit parameters')
    ax2 = ax1.twinx()

    ax1.set_xlabel(r'$\alpha$')
    ax1.set_xlim(np.min(self.x_alpha), np.max(self.x_alpha))

    leg1, = ax1.plot(self.x_alpha, self._a, marker='o', markersize=5, linestyle=None, color='navy', label=r'$a$')
    ax1.fill_between(self.x_alpha, y1=self._a-self._a_err, y2=self._a+self._a_err, color='lightblue', alpha=0.3)
    ax1.hlines(-1, np.min(self.x_alpha), np.max(self.x_alpha), color='grey', linewidth=0.5)
    ax1.set_ylabel(r'exponent $a$')

    leg2, = ax2.plot(self.x_alpha, self._A, marker='o', markersize=5, linestyle=None, color='forestgreen', label=r'$A$')
    ax2.fill_between(self.x_alpha, y1=self._A-self._A_err, y2=self._A+self._A_err, color='lightgreen', alpha=0.3)
    ax2.set_ylabel(r'prefactor $A$')

    plt.legend(handles=[leg1, leg2])

    plt.savefig('plots/alpha_'+self.name+'.png', format='png',dpi=300)


def plot_validity(self):
    '''
    Plots the distribution of invalid states.
    '''
    plt.figure(figsize=(12, 9))
    plt.title(f'Invalidity distribution of estimates')

    height = np.sum(np.logical_not(self._valids), axis=0)
    plt.imshow(self._valids, cmap=colors.ListedColormap(['red', 'green']), alpha=0.4, aspect='auto')
    plt.plot(np.arange(0, self.N[2]), height, marker='o', markersize=5, color='red', label='number of invalids')

    plt.ylabel(r'index $N_{mean}$ axis/ total numbe of invalids')
    plt.xlabel(r'index $N_{ticks}$ axis')
    plt.xlim((-0.5, self.N[2]-0.5))
    plt.ylim((-0.5, self.N_mean-0.5))
    plt.legend()

    plt.savefig('plots/val_'+self.name+'.png', format='png', dpi=300)


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
