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
            assert all(notNone), 'Not all necessary parameters were initialized.'

            # initialize other attributes
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
            f0 = lambda n, a, A: A * n**a
            f1 = lambda n, a, A: A * n**a
        else:
            f0 = lambda n, a, A: A * n**a
            f1 = lambda n, a, A: A * (n-self.d['N0'])**a

        try:
            if self.d['cup']:
                popt_0, pcov_0 = curve_fit(f0, self.x_N[:idx_N0], mean[:idx_N0], sigma=std[:idx_N0])
                popt_0_err = np.sqrt(np.diag(pcov_0))
            else:
                popt_0, pcov_0 = curve_fit(f0, self.x_N[:idx_N0], mean[:idx_N0], p0=[-0.5, 0.2], sigma=std[:idx_N0])
                popt_0_err = np.sqrt(np.diag(pcov_0))

            self.logger.info('curve_fit successful!')
            self.logger.info(f"\n"\
                'Fit parameters\n'\
                '--------------\n'\
                'scaling first step:\n'\
                f"a = {popt_0[0]:.2e} +/- {np.sqrt(popt_0_err[0]):.2e}\n"\
                f"A = {popt_0[1]:.2e} +/- {np.sqrt(popt_0_err[1]):.2e}")
        except Exception as e:
            self.logger.debug('The following error occurred in extract_fitparam: '+str(e))
            self.logger.info('curve_fit in first step not successful!')

            popt_0     = [None, None]
            popt_0_err = [None, None]

        try:
            if self.d['cup']:
                popt_1, pcov_1 = curve_fit(f1, self.x_N[idx_N0:], mean[idx_N0:], maxfev=4000)
                popt_1_err = np.sqrt(np.diag(pcov_1))
            else:
                popt_1, pcov_1 = curve_fit(f1, self.x_N[idx_N0:], mean[idx_N0:], p0=[-0.7, 3], maxfev=4000)
                popt_1_err = np.sqrt(np.diag(pcov_1))

            self.logger.info('curve_fit successful!')
            self.logger.info(f"\n"\
                'Fit parameters\n'\
                '--------------\n'\
                'scaling second step:\n'\
                f"a = {popt_1[0]:.2e} +/- {np.sqrt(popt_1_err[0]):.2e}\n"\
                f"A = {popt_1[1]:.2e} +/- {np.sqrt(popt_1_err[1]):.2e}")
        except Exception as e:
            self.logger.debug('The following error occurred in extract_fitparam: '+str(e))
            self.logger.info('curve_fit in second step not successful!')

            popt_1     = [None, None]
            popt_1_err = [None, None]

        f        = [f0, f1]
        popt     = [popt_0, popt_1]
        popt_err = [popt_0_err, popt_1_err]

        return f, popt, popt_err


    def calculate_scaling(self):
        try:
            _, popt, pcov = self.extract_fitparam()

            a = popt[:,0]
            A = popt[:,1]
            a_err = pcov[:,0]
            A_err = pcov[:,1]

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



class TwoStepAlpha(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name = name
        self.path = path

        # setup logging
        self.debug = debug
        self.setup_logging('TSA - '+self.name)

        if new:
            assert d is not None, 'Dictionary needs to be specified to build new class.'

            # add new parameters
            self.d = {}
            self.d['dim']         = None
            self.d['N_min']       = None
            self.d['N_max']       = None
            self.d['N_ticks']     = None
            self.d['N_mean']      = None
            self.d['alpha_min']   = None
            self.d['alpha_max']   = None
            self.d['alpha_ticks'] = None
            self.d['mirror']      = None
            self.d['cup']         = None
            self.d['f_N0']        = None
            self.d['f_sample']    = None
            self.d['f_estimate']  = None
            self.d['f_distance']  = None

            # initialize dictionary
            for key in d.keys():
                self.d[key] = d[key]

            notNone = [v is not None for v in self.d.values()]
            assert all(notNone), 'Not all necessary parameters were initialized.'

            # initialize other attributes
            self.povm    = const.povm[self.d['povm_name']]
            self.x_alpha = np.linspace(self.d['alpha_min'], self.d['alpha_max'], self.d['alpha_ticks'], endpoint=False, dtype=np.float)

            # initialize list where model will be stored in
            self._list = []

            # initialize storage for results
            self._originals = None
            self._estimates = np.empty((self.d['alpha_ticks'], self.d['N_mean'], self.d['N_ticks'], self.d['dim'], self.d['dim']), dtype=np.complex)
            self._valids    = np.ones((self.d['alpha_ticks'], self.d['N_mean'], self.d['N_ticks']), dtype=bool)
            self._distances = np.empty((self.d['alpha_ticks'], self.d['N_mean'], self.d['N_ticks']), dtype=np.float)

            self._a     = np.zeros((3, self.d['alpha_ticks']), dtype=np.float)
            self._a_err = np.zeros((3, self.d['alpha_ticks']), dtype=np.float)
            self._A     = np.zeros((3, self.d['alpha_ticks']), dtype=np.float)
            self._A_err = np.zeros((3, self.d['alpha_ticks']), dtype=np.float)
        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                tst = pickle.load(file)

            self.d       = tst.d
            self.x_alpha = tst.x_alpha
            self.povm    = tst.povm

            self._list = tst._list

            self._originals = tst._originals
            self._estimates = tst._estimates
            self._valids    = tst._valids
            self._distances = tst._distances

            self._a     = tst._a
            self._a_err = tst._a_err
            self._A     = tst._A
            self._A_err = tst._A_err

            self.logger.info('Loading already existing estimation data!')

        # report loaded parameters
        self.parameter_report()


    def get_list(self):
        return self._list

    def get_a(self):
        return self._a

    def get_A(self):
        return self._A


    def create_models(self):
        for idx, alpha in enumerate(self.x_alpha):
            name = self.name+f'_alpha{idx}'

            tst_keys       = ['dim', 'N_min', 'N_max', 'N_ticks', 'N_mean', 'povm_name', 'mirror', 'cup', 'f_N0', 'f_sample', 'f_estimate', 'f_distance']
            tst_d          = {key: self.d[key] for key in tst_keys}
            tst_d['alpha'] = alpha

            self._list.append(TwoStepTomography(name, self.path, True, self.debug, tst_d))

            N0 = self.d['f_N0'](self.d['N_max'], alpha)
            x0 = np.logspace(np.log10(self.d['N_min']), np.log10(N0), 5, dtype=np.int)
            x1 = N0 + np.logspace(np.maximum(np.log10(self.d['N_min']), 2), np.log10(self.d['N_max']-N0), self.d['N_ticks']-5, dtype=np.int)
            self._list[-1].x_N = np.concatenate((x0, x1))

            self._list[-1].create_originals()
            self._list[-1].reconstruct()

            try:
                _, popt, popt_err = self._list[-1].extract_fitparam()
                a_avg, A_avg, a_avg_err, A_avg_err = self._list[-1].get_scaling()

                self._a[:,idx]     = [popt[0][0], popt[1][0], a_avg]
                self._A[:,idx]     = [popt[0][1], popt[1][1], A_avg]
                self._a_err[:,idx] = [popt_err[0][1], popt_err[1][1], a_avg_err]
                self._A_err[:,idx] = [popt_err[0][1], popt_err[1][1], A_avg_err]
            except Exception as e:
                # logger information
                self.logger.debug('The following error occurred in create_models: '+str(e))

            self.logger.info(f"Tomography model for alpha = {alpha} initialized.")


    def review_fit(self, idx_alpha):
        try:
            _, popt, popt_err = self._list[idx_alpha].extract_fitparam()
            a_avg, A_avg, a_avg_err, A_avg_err = self._list[idx_alpha].get_scaling()

            self.logger.debug(self._a[:,idx_alpha])
            self.logger.debug([popt[0][0], popt[1][0], a_avg])

            self._a[:,idx_alpha]     = [popt[0][0], popt[1][0], a_avg]
            self._A[:,idx_alpha]     = [popt[0][1], popt[1][1], A_avg]
            self._a_err[:,idx_alpha] = [popt_err[0][1], popt_err[1][1], a_avg_err]
            self._A_err[:,idx_alpha] = [popt_err[0][1], popt_err[1][1], A_avg_err]

            self._list[idx_alpha].plot_distance()
        except Exception as e:
            # logger information
            self.logger.debug('The following error occurred in review_fit: '+str(e))
            self._list[idx_alpha].plot_distance()

    def plot_alpha_dependency(self):
        visualization.plot_alpha_dependency(self)
