from tomography import Tomography
from tomography import Comparison
from scipy.optimize import curve_fit
import numpy as np
import pickle
import logging

import general
import check
import const
import simulate
import visualization


class TwoStepTomography(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path

        # setup logging
        self.debug = debug
        self.setup_logging('TST - '+name)

        if new:
            assert d is not None, 'Want to build up model from scratch but certain variables are not specified.'
            self.logger.info('Buildung from scratch.')

            # add new parameters
            self.d = {}
            self.d['dim']        = None
            self.d['N_min']      = None
            self.d['N_max']      = None
            self.d['N_ticks']    = None
            self.d['N_mean']     = None
            self.d['povm_name']  = None
            self.d['alpha']      = None
            self.d['mirror']     = None
            self.d['cup']        = None
            self.d['N0']         = None
            self.d['f_N0']       = None
            self.d['f_sample']   = None
            self.d['f_estimate'] = None
            self.d['f_distance'] = None

            # initialize dictionary
            for key in d.keys():
                self.d[key] = d[key]
            self.d['N0'] = self.d['f_N0'](self.d['N_max'], self.d['alpha'])

            notNone = [v is not None for v in d.values()]
            assert all(notNone), 'Not all necessary parameters initialized.'

            # initialize other parameters
            self.povm    = const.povm[self.d['povm_name']]
            if self.d['cup']:
                x0       = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N0']), int(self.d['N_ticks']/2), dtype=np.int)
                x1       = np.logspace(np.log10(self.d['N0']), np.log10(self.d['N_max']), self.d['N_ticks'] - int(self.d['N_ticks']/2), dtype=np.int)
                self.x_N = np.concatenate((x0, x1))
            else:
                x0       = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N0']), int(self.d['N_ticks']/2), dtype=np.int)
                x1       = self.d['N0'] + np.logspace(np.maximum(np.log10(self.d['N_min']), 2), np.log10(self.d['N_max']-self.d['N0']), self.d['N_ticks'] - int(self.d['N_ticks']/2), dtype=np.int)
                self.x_N = np.concatenate((x0, x1))

            # initialize storage for results
            self._originals = None
            self._estimates = np.empty((self.d['N_mean'], self.d['N_ticks'], self.d['dim'], self.d['dim']), dtype=np.complex)
            self._valids    = np.ones((self.d['N_mean'], self.d['N_ticks']), dtype=bool)
            self._distances = np.empty((self.d['N_mean'], self.d['N_ticks']), dtype=np.float)

            self._a     = None
            self._a_err = None
            self._A     = None
            self._A_err = None
        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                tst = pickle.load(file)

            self.d    = tst.d
            self.x_N  = tst.x_N
            self.povm = tst.povm

            self._originals = tst._originals
            self._estimates = tst._estimates
            self._valids    = tst._valids
            self._distances = tst._distances

            try:
                self._a     = tst._a
                self._a_err = tst._a_err
                self._A     = tst._A
                self._A_err = tst._A_err
            except:
                self.logger.info('No optimal scaling data available!')

            self.logger.info('Loading already existing estimation data!')

        # report loaded parameters
        self.parameter_report()


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f"There are no sample states to reconstruct from."

        for j in range(self.d['N_mean']):
            self.logger.info(f"{j} of {self.d['N_mean']} states reconstructed.")
            for i in range(self.d['N_ticks']):
                self._estimates[j,i] = self.d['f_estimate'](self._originals[j], self.povm, self.x_N[i], self.d['N0'], cup=self.d['cup'], mirror=self.d['mirror'])
                self._distances[j,i] = self.d['f_distance'](self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])

        self.calculate_scaling()

        self.logger.info(f"score of valid states: {np.sum(self._valids)/(self.d['N_mean']*self.d['N_ticks'])}")


    def extract_fitparam(self):
        idx_N0 = np.argmax(self.x_N>self.d['N0'])

        mean = np.mean(self._distances, axis=0, where=self._valids)
        std  = np.std(self._distances, axis=0, where=self._valids)

        if self.d['cup']:
            try:
                f = lambda n, a, A: A * n**a

                popt_0, pcov_0 = curve_fit(f, self.x_N[:idx_N0], mean[:idx_N0], p0=[-0.5, 0.2], sigma=std[:idx_N0])
                popt_1, pcov_1 = curve_fit(f, self.x_N[idx_N0:], mean[idx_N0:], p0=[-0.8, 1])

                popt     = np.concatenate((popt_0[:,None], popt_1[:,None]), axis=1)
                popt_err = np.concatenate((np.diag(np.sqrt(pcov_0))[:,None], np.diag(np.sqrt(pcov_1))[:,None]), axis=1)

                self.logger.info('curve_fit successful!')
                self.logger.info(f"\n"\
                    'Fit parameters\n'\
                    '--------------\n'\
                    'scaling first step:\n'\
                    f"a = {popt[0,0]:.2e} +/- {np.sqrt(popt_err[0,0]):.2e}\n"\
                    f"A = {popt[1,0]:.2e} +/- {np.sqrt(popt_err[1,0]):.2e}\n"\
                    'scaling second step:\n'\
                    f"a = {popt[0,1]:.2e} +/- {np.sqrt(popt_err[0,1]):.2e}\n"\
                    f"A = {popt[1,1]:.2e} +/- {np.sqrt(popt_err[1,1]):.2e}")

                return f, popt, popt_err
            except Exception as e:
                self.logger.info('curve_fit not successful!')
                self.logger.debug('The following error occurred in extract_fitparam: '+str(e))
                return f, None, None
        else:
            try:
                f = lambda n, a, A: A * n**a

                popt_0, pcov_0 = curve_fit(f, self.x_N[:idx_N0], mean[:idx_N0], p0=[-0.5, 0.2], sigma=std[:idx_N0])
                popt_1, pcov_1 = curve_fit(f, self.x_N[idx_N0+5:]-self.d['N0'], mean[idx_N0+5:], p0=[-0.7, 3], maxfev=4000)

                popt     = np.concatenate((popt_0[:,None], popt_1[:,None]), axis=1)
                popt_err = np.concatenate((np.diag(np.sqrt(pcov_0))[:,None], np.diag(np.sqrt(pcov_1))[:,None]), axis=1)

                self.logger.info('curve_fit successful!')
                self.logger.info(f"\n"\
                    'Fit parameters\n'\
                    '--------------\n'\
                    'scaling first step:\n'\
                    f"a = {popt[0,0]:.2e} +/- {np.sqrt(popt_err[0,0]):.2e}\n"\
                    f"A = {popt[1,0]:.2e} +/- {np.sqrt(popt_err[1,0]):.2e}\n"\
                    'scaling shifted second step:\n'\
                    f"a = {popt[0,1]:.2e} +/- {np.sqrt(popt_err[0,1]):.2e}\n"\
                    f"A = {popt[1,1]:.2e} +/- {np.sqrt(popt_err[1,1]):.2e}")

                return f, popt, popt_err
            except Exception as e:
                self.logger.info('curve_fit not successful!')
                self.logger.debug('The following error occurred in extract_fitparam: '+str(e))


    def calculate_scaling(self):
        try:
            f, [a, A], [a_err, A_err] = self.extract_fitparam()

            if self.d['cup']:
                self._a     = np.log(A[1]/A[0])/np.log(self.d['N_max']) + a[1]
                self._a_err = np.sqrt( (1/(np.log(self.d['N_max'])*A[1]) * A_err[1])**2 + (1/(np.log(self.d['N_max'])*A[0]) * A_err[0])**2 + (a_err[1])**2 )
                self._A     = A[0]
                self._A_err = A_err[0]
            else:
                self._a     = np.log(A[1]/A[0])/np.log(self.d['N_max']) + a[1] * np.log(self.d['N_max'] - self.d['N0'])/np.log(self.d['N_max'])
                self._a_err = np.sqrt( (1/(np.log(self.d['N_max'])*A[1]) * A_err[1])**2 + (1/(np.log(self.d['N_max'])*A[0]) * A_err[0])**2 + (np.log(self.d['N_max'] - self.d['N0'])/np.log(self.d['N_max']) * a_err[1])**2 )
                self._A     = A[0]
                self._A_err = A_err[0]

            self.logger.info(f"\n"\
                'Fit parameters overall scaling\n'\
                '------------------------------\n'\
                f"a = {self._a:.2e} +/- {self._a_err:.2e}\n"\
                f"A = {self._A:.2e} +/- {self._A_err:.2e}")
        except Exception as e:
            self.logger.info("Extracting parameters of overall scaling wasn't successful")
            self.logger.debug('The following error occurred in calculate_scaling: '+str(e))



class TwoStepComparison(Comparison):

    def __init__(self, name, path, new, debug, name_list):

        self.name = name
        self.path = path
        self.new  = new

        # logging
        self.debug = debug
        self.setup_logging('TSC - '+self.name)

        # load data into
        self._list = []
        for name in name_list:
            self._list.append(TwoStepTomography(name, self.path, False, self.debug))
            self.logger.info(self._list[-1].parameter_report())

        tst_ref = self._list[0]

        # check comparison
        assert all([tst_ref.d['dim'] == tst.d['dim'] for tst in self._list]), 'Different dimension encountered. Comparison does not make sense!'
        # assert all([tst_ref.d['N_mean'] == tst.d['N_mean'] for tst in self._list]), 'Different N_mean encountered. Comparison does not make sense!'
        assert all([tst_ref.d['N_max'] == tst.d['N_max'] for tst in self._list]), 'Different N_max encountered. Comparison does not make sense!'
        assert all([tst_ref.d['f_sample'] == tst.d['f_sample'] for tst in self._list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([tst_ref.d['f_distance'] == tst.d['f_distance'] for tst in self._list]), 'Different distance measures encountered. Comparison does not make sense!'

        self.d = {}
        self.d['dim']        = tst_ref.d['dim']
        self.d['N_mean']     = tst_ref.d['N_mean']
        self.d['N_max']      = tst_ref.d['N_max']
        self.d['f_sample']   = tst_ref.d['f_sample']
        self.d['f_distance'] = tst_ref.d['f_distance']


    def get_alpha(self):
        return [tst.d['alpha'] for tst in self._list]

    def get_mirror(self):
        return [tst.d['mirror'] for tst in self._list]

    def get_cup(self):
        return [tst.d['cup'] for tst in self._list]


    def transform_citeria(self, criteria):

        data = {}
        data['f_estimate'] = [visualization.w[f_estimate] for f_estimate in self.get_estimation_method()]
        data['povm_name']  = self.get_povm_name()
        data['alpha']      = [fr"$\alpha$ = {alpha}" for alpha in self.get_alpha()]
        data['mirror']     = ['aligned' if mirror else 'anti-aligned' for mirror in self.get_mirror()]
        data['cup']        = ["$D_0\cup D_1$" if cup else "$D_1$" for cup in self.get_cup()]

        return data[criteria]


class AlphaDependency(Tomography):

    x_alpha = None

    _list = []

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path
        self.new   = new

        # setup logging
        self.debug = debug
        self.setup_logging('AD - '+self.name)

        # add new paraemters
        self.d['alpha_min']   = None
        self.d['alpha_max']   = None
        self.d['alpha_ticks'] = None
        self.d['mirror']      = None
        self.d['f_N0']        = None

        # reload data
        if self.new:
            assert d is not None, 'Want to build up model from scratch but certain variables are not specified.'
            self.logger.info('Buildung from scratch.')

            for key in d.keys():
                self.d[key] = d[key]

            self.x_N     = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), self.d['N_ticks'], dtype=np.int)
            self.x_alpha = np.linspace(self.d['alpha_min'], self.d['alpha_max'], self.d['alpha_ticks'])
            self.povm    = const.povm[self.d['povm_name']]

            self._a          = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._a_err      = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._A          = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._A_err      = np.zeros(self.d['alpha_ticks'], dtype=np.float)

            notNone = [v is not None for v in d.values()]
            assert all(notNone), 'Not all necessary parameters initialized.'
        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                self.logger.info('Loading already existing estimation data!')
                ad = pickle.load(file)

            self.d = ad.d

            self._alpha_list = ad._alpha_list
            self._a          = ad._a
            self._a_err      = ad._a_err
            self._A          = ad._A
            self._A_err      = ad._A_err


        # report loaded parameters
        self.parameter_report()


    def get_list(self):
        return self._list

    def get_a(self):
        return self._a

    def get_A(self):
        return self._A


    def create_models(self):
        for alpha in self.x_alpha:
            name = self.name+'_'+str(np.round(alpha, decimals=8))

            tst_keys       = ['dim', 'N_min', 'N_max', 'N_ticks', 'N_mean', 'x_n', 'mirror', 'povm_name', 'f_sample', 'f_estimate', 'f_distance', 'f_N0']
            tst_d          = {key: self.d[key] for key in tst_keys}
            tst_d['alpha'] = alpha

            self._list.append(TwoStepTomography(name, True, self.debug, tst_d))

            self._list[-1].create_originals()
            self._list[-1].reconstruct()

            self.logger.info(f"Tomography model for alpha = {alpha} initialized.")


    def extract_gradient(self):
        for idx, tomo in enumerate(self._list):

            # calculate mean
            mean = np.mean(tomo.get_distances(), axis=0, where=tomo.get_valids())
            std  = np.std(tomo.get_distances(), axis=0, where=tomo.get_valids())

            # fit
            try:
                f = lambda x, a, A: A * (x - tomo.d['alpha']*x)**a
                popt, pcov = curve_fit(f, tomo.x_N, mean, p0=[-0.5, 1], sigma=std)

                self._a[idx], self._A[idx]         = popt
                self._a_err[idx], self._A_err[idx] = np.sqrt(np.diag(pcov))

                self.logger.info('curve_fit successful!')
            except:
                 self.logger.info('curve_fit not successful!')


    def review_fit(self, idx_alpha):
        self._list[idx_alpha].plot_distance()

    def compare_fit(self, list_idx_alpha):
        pass

    def plot_alpha_dependency(self):
        visualization.plot_alpha_dependency(self)
