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
import state
import check
import align

# name dictionary for functions
w = {}

# f_sample
w[pure.sample_unitary]    = 'pure'
w[mixed.sample_bures]     = 'mixed'
w[mixed.sample_hilbert]   = 'mixed'
w[pure.sample_product_unitary]  = 'pure product'
w[mixed.sample_product_bures]   = 'mixed product'
w[mixed.sample_product_hilbert] = 'mixed product'

# states
w[state.bell1] = 'Bell-1'
w[state.bell2] = 'Bell-2'
w[state.bell3] = 'Bell-3'
w[state.bell4] = 'Bell-4'

# f_align
w[align.eigenbasis]         = 'non product U'
w[align.product_eigenbasis] = 'product U'

# f_estimate
w[mle.iterative]      = 'MLE'
w[mle.two_step]       = 'two step MLE'
w[inversion.linear]   = 'LI'
w[inversion.two_step] = 'two step LI'

# f_distance
w[general.euclidean_dist]  = 'Euclidean distance'
w[general.hilbert_dist]    = 'Hilbert Schmidt distance'
w[general.infidelity]      = 'infidelity'


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
        except Exception as e:
            b.add_points(general.expect_xyz(points))
            print(f"The follwoing exception occurred in qubit: {e}")
        b.render()
    if np.all(vectors!= None):
        try:
            b.add_vectors(general.expect_xyz(vectors))
        except Exception as e:
            b.add_vectors(general.expect_xyz(vectors))
            print(f"The follwoing exception occurred in qubit: {e}")
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
    b.vector_color = ['blue', *np.repeat('red', len(rho_1[1])), *np.repeat('violet', len(rho_2[1]))]
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
            # b.add_points(pnt)
            b.add_vectors(pnt)

        rho = rho_1[1][-1]
        vec = general.expect_xyz(rho)
        b.add_vectors(vec)
        b.add_annotation(vec/2, rho_1[0], c='red', fontsize=10)
        b.render()

    if rho_2[0]!=None:
        for i in range(len(rho_2[1])-1):
            rho = rho_2[1][i]
            pnt = general.expect_xyz(rho)
            # b.add_points(pnt)
            b.add_vectors(pnt)

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
    Visualizes the eigenvalue distribution of an array of states.

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


def purity_distribution(rhos: dict):
    '''
    Visualizes the purity distribution of the given states.

    :param rhos: dictionary
        rhos["description"] = Nxdxd array of states
    :return:
    '''
    plt.figure(figsize=(15, 9))
    plt.title('Purity distribution')

    for key in rhos.keys():
        rhos_red = general.partial_trace(rhos[key], 0)
        plt.hist(np.real(general.purity(rhos_red)), int(np.sqrt(len(rhos_red))), density=False, color='navy', alpha=0.3, label=key)

    plt.xlabel('purity')
    plt.ylabel(r'$P$')
    plt.xlim(1/2, 1)
    plt.legend()

    plt.show()


def plot_distance1(self, n=0):
    '''
    Plots the N-dependency of the distance measure for the given simulation.

    :param n: integer for disregarding the first n measurement points
    '''
    idx_N0 = np.argmax(self.x_N>self.d['N0'])

    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {w[self.d['f_sample']]} {int(np.log2(self.d['dim']))} qubit states")

    # calculate mean
    mean = np.mean(self.get_distances(), axis=0, where=self.get_valids())
    std  = np.std(self.get_distances(), axis=0, where=self.get_valids())

    # plot measurement data
    plt.plot(self.x_N[:idx_N0], mean[:idx_N0], color='navy', linestyle='None', markersize=5, marker='o', label=f"frist step {w[self.d['f_estimate']]}")
    plt.plot(self.x_N[idx_N0:], mean[idx_N0:], color='forestgreen', linestyle='None', markersize=5, marker='o', label=f"second step {w[self.d['f_estimate']]}")
    plt.axvline(self.d['N0'], color='grey', linewidth=0.5, label='two-step threshold')

    # plot fits
    f, popt, popt_err = self.calculate_fitparam(n=n)
    x  = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), 100, dtype=np.int32)
    x0 = x[x<=self.d['N0']]
    x1 = x[x>self.d['N0']]

    # plot first part
    try:
        plt.fill_between(x0, y1=f[0](x0, *(popt[0]-popt_err[0])), y2=f[0](x0, *(popt[0]+popt_err[0])), color='lightblue', alpha=0.3)
        plt.plot(x, f[0](x, *popt[0]), color='lightblue', label=f"scaling first step: a = {popt[0][0]:.2e} $\pm$ {popt_err[0][0]:.2e}")
    except Exception as e:
        self.logger.info(f"Plotting first fit wasn't successful!")
        self.logger.debug('The following error occurred in plot_distance: '+str(e))

    # plot second part
    try:
        if self.d['cup']:
            plt.fill_between(x1, y1=f[1](x1, *(popt[1]-popt_err[1])), y2=f[1](x1, *(popt[1]+popt_err[1])), color='lightgreen', alpha=0.3)
            plt.plot(x, f[1](x, *popt[1]), color='lightgreen', label=f"scaling second step: a = {popt[1][0]:.2e} $\pm$ {popt_err[1][0]:.2e}")
        else:
            # plot shifted results
            plt.plot(self.x_N[idx_N0:]-self.d['N0'], mean[idx_N0:], color='darkorange', linestyle='None', markersize=5, marker='o', label=f"shifted second step {w[self.d['f_estimate']]} with {self.d['povm_name']}")

            self.logger.debug(f"{popt[1]}{popt_err[1]}")
            plt.fill_between(x, y1=f[1](x+self.d['N0'], *(popt[1]-popt_err[1])), y2=f[1](x+self.d['N0'], *(popt[1]+popt_err[1])), color='bisque', alpha=0.3)
            plt.plot(x, f[1](x+self.d['N0'], *popt[1]), color='bisque', label=f"scaling shifted second step: a = {popt[1][0]:.2e} $\pm$ {popt_err[1][0]:.2e}")
    except Exception as e:
        self.logger.info(f"Plotting second fit wasn't successful!")
        self.logger.debug('The following error occurred in plot_distance: '+str(e))

    # plot average
    try:
        a, A, a_err, A_err = self.get_scaling()
        plt.plot(x, f[0](x, a[2], A[2]), color='red', linewidth=0.5, label=f"overall scaling: a {a[2]:.2e} $\pm$ {a_err[2]:.2e}")
    except Exception as e:
        self.logger.info(f"Plotting average fit wasn't successful!")
        self.logger.debug('The following error occurred in plot_distance: '+str(e))

    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(self.x_N[0], self.x_N[-1])
    plt.legend()

    plt.savefig(self.path+'plots/dist_'+self.name+'.png', format='png', dpi=300)


def plot_distance2(self, n=0):
    '''
    Plots distance for OneStepTomography and TwoStepTomography2.

    :param n: integer for disregarding the first n measurement points
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {w[self.d['f_sample']]} {int(np.log2(self.d['dim']))} qubit states")

    # calculate mean
    mean = np.mean(self.get_distances(), axis=0, where=self.get_valids())
    std  = np.std(self.get_distances(), axis=0, where=self.get_valids())

    # plot measurement data
    plt.plot(self.x_N, mean, color='navy', linestyle='None', markersize=5, marker='o', label=f"{w[self.d['f_estimate']]}")

    f, popt, popt_err = self.calculate_fitparam(n=n)
    x = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), 100, dtype=np.int)
    try:
        plt.fill_between(x, y1=f(x, *(popt-popt_err)), y2=f(x, *(popt+popt_err)), color='lightblue', alpha=0.3)
        plt.plot(x, f(x, *popt), color='lightblue', label=f"scaling: a = {popt[0]:.2e} $\pm$ {popt_err[0]:.2e}")
    except Exception as e:
        self.logger.info(f"Plotting fit wasn't successful!")
        self.logger.debug('The following error occurred in plot_distance2: '+str(e))

    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(self.x_N[0], self.x_N[-1])
    plt.legend()

    plt.savefig(self.path+'plots/dist_'+self.name+'.png', format='png', dpi=300)


def compare_distance_osc(self, criteria_1, criteria_2):
    '''
    Compares up to four different Tomography schemes based on up to two criteria in one plot.

    :param criteria_1: first criteria need to be consideread, same order as self.tomo_list
    :param criteria_2: second criteria need to be considered, same order as self.tomo_list
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {w[self.d['f_sample']]} {int(np.log2(self.d['dim']))} qubit states")

    c = [['navy', 'lightblue'], ['forestgreen', 'lightgreen'], ['red', 'lightsalmon'], ['black', 'grey'], ['peru', 'sandybrown'], ['darkorange', 'bisque']]
    for idx, tomo in enumerate(self._list):

        # calculate mean
        mean = np.mean(tomo.get_distances(), axis=0, where=tomo.get_valids())
        std  = np.std(tomo.get_distances(), axis=0, where=tomo.get_valids())

        # plot both steps
        plt.plot(tomo.x_N, mean, color=c[idx][0], linestyle='None', markersize=5, marker='o', label=fr"{w[tomo.d['f_estimate']]} with {criteria_1[idx]} and {criteria_2[idx]}")

        # fit and plot overall curve
        f = lambda x, a, A: A*x**a
        a, A, a_err, _ = tomo.get_scaling()

        x  = np.logspace(np.log10(tomo.d['N_min']), np.log10(tomo.d['N_max']), 100, dtype=np.float)
        plt.plot(x, f(x, a, A), color=c[idx][1], linewidth=1, label=f"scaling: a {a:.2e} $\pm$ {a_err:.2e}")

    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((np.min(self.get_N_min()), self.d['N_max']))
    plt.legend()

    plt.savefig(self.path+'plots/comp_'+self.name+'.png', format='png', dpi=300)


def compare_distance(self, criteria_1, criteria_2):
    '''
    Compares up to four different Tomography schemes based on up to two criteria in one plot.

    :param criteria_1: first criteria need to be consideread, same order as self.tomo_list
    :param criteria_2: second criteria need to be considered, same order as self.tomo_list
    '''
    # initialize plot
    plt.figure(figsize=(12, 9))
    plt.title(f"N-scaling of {w[self.d['f_distance']]} averaged over {self.d['N_mean']} {int(np.log2(self.d['dim']))} qubit states")

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
            a, A, a_err, A_err = tomo.get_scaling()
            a, A, a_err, A_err = a[-1], A[-1], a_err[-1], A_err[-1]

            x  = np.logspace(np.log10(tomo.d['N_min']), np.log10(tomo.d['N_max']), 100, dtype=np.int32)
            plt.plot(x, f(x, a, A), color=c[idx][1], linewidth=1, label=f"overall scaling: a {a:.2e} $\pm$ {a_err:.2e}")

            self.logger.info(f"Plotting fit was successful!")
        except Exception as e:
            self.logger.info(f"Plotting fit wasn't successful!")
            self.logger.debug('The following error occurred in compare_distance: '+str(e))


    plt.xlabel(r'$N$')
    plt.ylabel(f"{w[self.d['f_distance']]}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((np.min(self.get_N_min()), self.d['N_max']))
    plt.legend()

    plt.savefig(self.path+'plots/comp_'+self.name+'.png', format='png', dpi=300)


def plot_a_dependency1(self, error=False):
    '''
    Plots the alpha dependency of the scaling.
    '''
    plt.figure(figsize=(12,9))
    plt.title(r'$\alpha$-dependency of the scaling in TST1')

    if error:
        plt.fill_between(self.x_alpha, y1=self._a[:,0]-self._a_err[:,0], y2=self._a[:,0]+self._a_err[:,0], color='navy', alpha=0.3)
        plt.fill_between(self.x_alpha, y1=self._a[:,1]-self._a_err[:,1], y2=self._a[:,1]+self._a_err[:,1], color='forestgreen', alpha=0.3)
        plt.fill_between(self.x_alpha, y1=self._a[:,2]-self._a_err[:,2], y2=self._a[:,2]+self._a_err[:,2], color='red', alpha=0.3)

    plt.plot(self.x_alpha, self._a[:,0], marker='o', markersize=5, color='navy', label=r'$a_0$')
    plt.plot(self.x_alpha, self._a[:,1], marker='o', markersize=5, color='forestgreen', label=r'$a_1$')
    plt.plot(self.x_alpha, self._a[:,2], marker='o', markersize=5, color='red', label=r'$a_{avg}$')

    plt.xlim(np.min(self.x_alpha), np.max(self.x_alpha))
    plt.ylabel(r'exponent $a$')
    plt.xlabel(r'$\alpha$')
    plt.legend()

    plt.savefig(self.path+'plots/alpha_scaling_'+self.name+'.png', format='png',dpi=300)


def plot_A_dependency1(self, error):
    '''
    Plots the alpha dependency of the intercept.
    '''
    plt.figure(figsize=(12,9))
    plt.title(r'$\alpha$-dependency of the intercept in TST1')

    if error:
        plt.fill_between(self.x_alpha, y1=self._A[:,2]-self._A_err[:,2], y2=self._A[:,2]+self._A_err[:,2], color='red', alpha=0.3)

    plt.plot(self.x_alpha, self._A[:,2], marker='o', markersize=5, color='red', label=r'$A_{avg}$')

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'intercept $A$')
    plt.xlim(np.min(self.x_alpha), np.max(self.x_alpha))
    plt.legend()

    plt.savefig(self.path+'plots/alpha_intercept_'+self.name+'.png', format='png',dpi=300)


def plot_a_dependency2(self, error=False):
    '''
    Plots the alpha dependency of the scaling.
    '''
    plt.figure(figsize=(12,9))
    plt.title(r'$\alpha$-dependency of the scaling in TST2')

    if error:
        plt.fill_between(self.x_alpha, y1=self._a-self._a_err, y2=self._a+self._a_err, color='lightblue', alpha=0.3)
    plt.plot(self.x_alpha, self._a, marker='o', markersize=5, color='navy', label=r'$a$')

    if self.d['f_N0']==general.N_exp:
        plt.axhline(-5/6, color='grey', linewidth=0.5)
        plt.axvline(2/3, color='grey', linewidth=0.5)
    elif self.d['f_N0']==general.N_frac:
        plt.axhline(-1, color='grey', linewidth=0.5)
        plt.axvline(1/2, color='grey', linewidth=0.5)

    plt.xlim(np.min(self.x_alpha), np.max(self.x_alpha))
    plt.ylabel(r'exponent $a$')
    plt.xlabel(r'$\alpha$')
    plt.legend()

    plt.savefig(self.path+'plots/alpha_scaling_'+self.name+'.png', format='png',dpi=300)


def plot_A_dependency2(self, error=False):
    '''
    Plots the alpha dependency of the intercept.
    '''
    plt.figure(figsize=(12,9))
    plt.title(r'$\alpha$-dependency of the intercept in TST2')

    if error:
        plt.fill_between(self.x_alpha, y1=self._A-self._A_err, y2=self._A+self._A_err, color='lightblue', alpha=0.3)
    plt.plot(self.x_alpha, self._A, marker='o', markersize=5, color='navy', label=r'$a_0$')

    if self.d['f_N0']==general.N_frac:
        plt.axvline(1/2, color='grey', linewidth=0.5)

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'intercept $A$')
    plt.xlim(np.min(self.x_alpha), np.max(self.x_alpha))
    plt.legend()

    plt.savefig(self.path+'plots/alpha_intercept_'+self.name+'.png', format='png',dpi=300)


def plot_validity(self):
    '''
    Plots the distribution of invalid states.
    '''
    plt.figure(figsize=(12, 9))
    plt.title(f"Invalidity distribution of {w[self.d['f_sample']]} {int(np.log2(self.d['dim']))} qubit states")

    height = np.sum(np.logical_not(self._valids), axis=0)
    plt.imshow(self._valids, cmap=colors.ListedColormap(['red', 'green']), vmin=0, vmax=1, alpha=0.4, aspect='auto')
    plt.plot(np.arange(0, self.d['N_ticks']), height, marker='o', markersize=5, color='red', label='number of invalids')

    plt.ylabel(r'index $N_{mean}$ axis/ total numbe of invalids')
    plt.xlabel(r'index $N_{ticks}$ axis')
    plt.xlim((-0.5, self.d['N_ticks']-0.5))
    plt.ylim((-0.5, self.d['N_mean']-0.5))
    plt.legend()

    plt.savefig(self.path+'plots/val_'+self.name+'.png', format='png', dpi=300)


def speed_comparison(title, d, iterations=10):
    '''
    Shows the result of speed comparison of arbitrary functions.

    :param title     : title of the plot
    :param iterations: number of iterations the test function is tested
    :param **kwargs  : dictionary
        d[name] = (func, list of parameters)
    :return:
    '''
    data = speed.compare(d, iterations=iterations)
    print(data)
    df   = pd.DataFrame.from_dict(data, orient='index')

    ax = df.plot.bar(figsize=(10, 6), ylabel='time', title=title , legend=False, rot=0)
    ax.plot()

    plt.show()
