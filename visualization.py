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

# name dictionary for functions
w = {}

# f_sample
w[pure.unitary_to_density]    = 'pure'
w[mixed.sample_bures]         = 'mixed'
w[mixed.sample_hilbert]       = 'mixed'
w[mixed.hermitian_to_density] = 'mixed'

# f_estimate
w[mle.iterative]      = 'MLE'
w[mle.two_step]       = 'two step MLE'
w[inversion.linear]   = 'LI'
w[inversion.two_step] = 'two step LI'

# f_distance
w[general.euclidean_dist] = 'Euclidean distance'
w[general.hilbert_dist]   = 'Hilbert Schmidt distance'
w[general.infidelity]     = 'infidelity'


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
    plt.title(f"Eigenvalue distribution for sampling according to {m}")
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
    idx_N0 = np.argmax(self.x_N>self.d['N0'])

    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {w[self.d['f_sample']]} states")

    # calculate mean
    mean = np.mean(self.get_distances(), axis=0, where=self.get_valids())
    std  = np.std(self.get_distances(), axis=0, where=self.get_valids())

    # plot both steps
    plt.plot(self.x_N[:idx_N0], mean[:idx_N0], color='navy', linestyle='None', markersize=5, marker='o', label=f"frist step {w[self.d['f_estimate']]} with {self.d['povm_name']}")
    plt.plot(self.x_N[idx_N0:], mean[idx_N0:], color='forestgreen', linestyle='None', markersize=5, marker='o', label=f"second step {w[self.d['f_estimate']]} with {self.d['povm_name']}")
    plt.axvline(self.d['N0'], color='grey', linewidth=0.5, label='two-step threshold')

    # plot fit parameters
    if self.d['cup']:
        try:
            # plot fitted curves
            f, [a, A], [a_err, A_err] = self.extract_fitparam()

            x  = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), 100, dtype=np.int32)
            x0 = x[x<=self.d['N0']]
            x1 = x[x>self.d['N0']]

            plt.fill_between(x0, y1=f(x0, a[0]-a_err[0], A[0]-A_err[0]), y2=f(x0, a[0]+a_err[0], A[0]+A_err[0]), color='lightblue', alpha=0.3)
            plt.fill_between(x1, y1=f(x1, a[1]-a_err[1], A[1]-A_err[1]), y2=f(x1, a[1]+a_err[1], A[1]+A_err[1]), color='lightgreen', alpha=0.3)
            plt.plot(x, f(x, a[0], A[0]), color='lightblue', label=f"scaling first step: a = {a[0]:.2e} $\pm$ {a_err[0]:.2e}")
            plt.plot(x, f(x, a[1], A[1]), color='lightgreen', label=f"scaling second step: a = {a[1]:.2e} $\pm$ {a_err[1]:.2e}")

            # fit and plot overall curve
            a_avg, A_avg, a_avg_err, A_avg_err = self.get_scaling()

            plt.plot(x, f(x, a_avg, A_avg), color='red', linewidth=0.5, label=f"overall scaling: a {a_avg:.2e} $\pm$ {a_avg_err:.2e}")

            # logger information
            self.logger.info(f"Plotting fit was successful!")
        except Exception as e:
            # logger information
            self.logger.info(f"Plotting fit wasn't successful!")
            self.logger.debug('The following error occurred in plot_distance: '+str(e))
    else:
        try:
            # plot shifted results
            plt.plot(self.x_N[idx_N0:]-self.d['N0'], mean[idx_N0:], color='darkorange', linestyle='None', markersize=5, marker='o', label=f"shifted second step {w[self.d['f_estimate']]} with {self.d['povm_name']}")

            # plot fitted curves
            f, [a, A], [a_err, A_err] = self.extract_fitparam()

            x  = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), 100, dtype=np.int32)
            x0 = x[x<=self.d['N0']]
            x1 = x[x>self.d['N0']]

            plt.fill_between(x0, y1=f(x0, a[0]-a_err[0], A[0]-A_err[0]), y2=f(x0, a[0]+a_err[0], A[0]+A_err[0]), color='lightblue', alpha=0.3)
            plt.fill_between(x, y1=f(x, a[1]-a_err[1], A[1]-A_err[1]), y2=f(x, a[1]+a_err[1], A[1]+A_err[1]), color='bisque', alpha=0.3)
            plt.plot(x, f(x, a[0], A[0]), color='lightblue', label=f"scaling first step: a = {a[0]:.2e} $\pm$ {a_err[0]:.2e}")
            plt.plot(x, f(x, a[1], A[1]), color='bisque', label=f"scaling shifted second step: a = {a[1]:.2e} $\pm$ {a_err[1]:.2e}")

            # fit and plot overall curve
            a_avg, A_avg, a_avg_err, A_avg_err = self.get_scaling()

            plt.plot(x, f(x, a_avg, A_avg), color='red', linewidth=0.5, label=f"overall scaling: a {a_avg:.2e} $\pm$ {a_avg_err:.2e}")

            # logger information
            self.logger.info(f"Plotting fit was successful!")
        except Exception as e:
            # logger information
            self.logger.info(f"Plotting fit wasn't successful!")
            self.logger.debug('The following error occurred in plot_distance: '+str(e))

    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(self.x_N[0], self.x_N[-1])
    plt.legend()

    plt.savefig(self.path+'plots/dist_'+self.name+'.png', format='png', dpi=300)


def compare_distance(self, criteria_1, criteria_2):
    '''
    Compares up to four different Tomography schemes based on up to two criteria in one plot.

    :param criteria_1: first criteria need to be consideread, same order as self.tomo_list
    :param criteria_2: second criteria need to be considered, same order as self.tomo_list
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {w[self.d['f_sample']]} states")

    c = [['navy', 'lightblue'], ['forestgreen', 'lightgreen'], ['red', 'lightsalmon'], ['black', 'grey'], ['peru', 'sandybrown'], ['darkorange', 'bisque']]
    for idx, tomo in enumerate(self._list):

        # determine index
        idx_N0 = np.argmax(tomo.x_N>tomo.d['N0'])

        # calculate mean
        mean = np.mean(tomo.get_distances(), axis=0, where=tomo.get_valids())
        std  = np.std(tomo.get_distances(), axis=0, where=tomo.get_valids())

        # plot both steps
        plt.plot(tomo.x_N, mean, color=c[idx][0], linestyle='None', markersize=5, marker='o', label=fr"{w[tomo.d['f_estimate']]} with {criteria_1[idx]} and {criteria_2[idx]}")
        plt.axvline(tomo.d['N0'], color=c[idx][1], linewidth=0.5)

        try:
            # fit and plot overall curve
            f = lambda x, a, A: A*x**a
            a_avg, A_avg, a_avg_err, A_avg_err = tomo.get_scaling()

            x  = np.logspace(np.log10(tomo.d['N_min']), np.log10(tomo.d['N_max']), 100, dtype=np.int32)
            plt.plot(x, f(x, a_avg, A_avg), color=c[idx][1], linewidth=1, label=f"overall scaling: a {a_avg:.2e} $\pm$ {a_avg_err:.2e}")

            # logger information
            self.logger.info(f"Plotting fit was successful!")
        except Exception as e:
            # logger information
            self.logger.info(f"Plotting fit wasn't successful!")
            self.logger.debug('The following error occurred in compare_distance: '+str(e))


    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((np.min(self.get_N_min()), self.d['N_max']))
    plt.legend()

    plt.savefig(self.path+'plots/comp_'+self.name+'.png', format='png', dpi=300)


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

    plt.savefig(self.path+'plots/alpha_'+self.name+'.png', format='png',dpi=300)


def plot_validity(self):
    '''
    Plots the distribution of invalid states.
    '''
    plt.figure(figsize=(12, 9))
    plt.title(f"Invalidity distribution of estimates")

    height = np.sum(np.logical_not(self._valids), axis=0)
    plt.imshow(self._valids, cmap=colors.ListedColormap(['red', 'green']), alpha=0.4, aspect='auto')
    plt.plot(np.arange(0, self.d['N_ticks']), height, marker='o', markersize=5, color='red', label='number of invalids')

    plt.ylabel(r'index $N_{mean}$ axis/ total numbe of invalids')
    plt.xlabel(r'index $N_{ticks}$ axis')
    plt.xlim((-0.5, self.d['N_ticks']-0.5))
    plt.ylim((-0.5, self.d['N_mean']-0.5))
    plt.legend()

    plt.savefig(self.path+'plots/val_'+self.name+'.png', format='png', dpi=300)


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
