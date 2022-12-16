from tomography import Tomography
from tomography import Comparison
from scipy.optimize import curve_fit
import numpy as np
import pickle
import general
import check
import const
import inversion
import mle
import visualization


class TwoStepTomography1(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path

        # setup logging
        self.debug = debug
        self.setup_logging('TST1 - '+name)

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
            if self.d['cup']:
                x0       = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N0']), 4, dtype=np.int)
                x1       = np.logspace(np.log10(self.d['N0']), np.log10(self.d['N_max']), self.d['N_ticks']-4, dtype=np.int)
            else:
                x0       = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N0']), 4, dtype=np.int)
                x1       = self.d['N0'] + np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']-self.d['N0']), self.d['N_ticks']-4, dtype=np.int)
            self.x_N = np.concatenate((x0, x1))
            self.povm    = general.povm[self.d['povm_name']](self.d['dim'])

            # initialize storage for results
            self._originals = None
            self._estimates = np.empty((self.d['N_mean'], self.d['N_ticks'], self.d['dim'], self.d['dim']), dtype=np.complex)
            self._valids    = np.ones((self.d['N_mean'], self.d['N_ticks']), dtype=bool)
            self._distances = np.empty((self.d['N_mean'], self.d['N_ticks']), dtype=np.float)

            self._a     = np.empty(3, dtype=np.float)
            self._a_err = np.empty(3, dtype=np.float)
            self._A     = np.empty(3, dtype=np.float)
            self._A_err = np.empty(3, dtype=np.float)
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

            self._a     = tst._a
            self._a_err = tst._a_err
            self._A     = tst._A
            self._A_err = tst._A_err

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
        self.logger.info(f"score of valid states: {np.sum(self._valids)/(self.d['N_mean']*self.d['N_ticks'])}")
        self.calculate_fitparam()


    def calculate_fitparam(self, n=0):
        idx_N0 = np.argmax(self.x_N>self.d['N0'])
        assert idx_N0+n<self.d['N_ticks'], 'Parameter n is bigger than allowed!'

        # calculate means
        mean = np.mean(self._distances, axis=0, where=self._valids)
        std  = np.std(self._distances, axis=0, where=self._valids)

        # initialize fit functions
        if self.d['cup']:
            f0 = lambda n, a, A: A * n**a
            f1 = lambda n, a, A: A * n**a
        else:
            f0 = lambda n, a, A: A * n**a
            f1 = lambda n, a, A: A * (n-self.d['N0'])**a

        # fit first step
        try:
            if self.d['cup']:
                popt_0, pcov_0 = curve_fit(f0, self.x_N[:idx_N0], mean[:idx_N0])
                popt_0_err = np.sqrt(np.diag(pcov_0))
            else:
                popt_0, pcov_0 = curve_fit(f0, self.x_N[:idx_N0], mean[:idx_N0])
                popt_0_err = np.sqrt(np.diag(pcov_0))
            self.logger.info('first step curve_fit successful!')
        except Exception as e:
            self.logger.debug('The following error occurred in the first step of calculate_fitparam: '+str(e))
            self.logger.info('curve_fit in first step not successful!')
            popt_0     = [None, None]
            popt_0_err = [None, None]

        # fit second step
        try:
            if self.d['cup']:
                if self.d['f_estimate']==inversion.two_step:
                    popt_1     = [None, None]
                    popt_1_err = [None, None]
                else:
                    popt_1, pcov_1 = curve_fit(f1, self.x_N[idx_N0+n:], mean[idx_N0+n:])
                    popt_1_err = np.sqrt(np.diag(pcov_1))
            else:
                popt_1, pcov_1 = curve_fit(f1, self.x_N[idx_N0+n:], mean[idx_N0+n:])
                popt_1_err = np.sqrt(np.diag(pcov_1))
            self.logger.info('second step curve_fit successful!')
        except Exception as e:
            self.logger.debug('The following error occurred in the second step of calculate_fitparam: '+str(e))
            self.logger.info('curve_fit in second step not successful!')
            popt_1     = [None, None]
            popt_1_err = [None, None]

        # fit overall
        popt_2     = [None, None]
        popt_2_err = [None, None]

        if popt_0[0]!=None:
            popt_2[0] = 1/np.log(self.d['N_max']) * np.log(mean[-1]/popt_0[1])
            popt_2[1] = popt_0[1]
            popt_2_err[0] = 1/np.log(self.d['N_max']) * np.sqrt( (popt_0_err[1]/popt_0[1])**2 + (std[-1]/mean[-1])**2 )
            popt_2_err[1] = popt_0_err[1]

        # store fit parameters
        self._a = np.array([popt_0[0], popt_1[0], popt_2[0]])
        self._A = np.array([popt_0[1], popt_1[1], popt_2[1]])
        self._a_err = np.array([popt_0_err[0], popt_1_err[0], popt_2_err[0]])
        self._A_err = np.array([popt_0_err[1], popt_1_err[1], popt_2_err[1]])

        return [f0, f1], [popt_0, popt_1], [popt_0_err, popt_1_err]


    def plot_distance(self, n=0):
        visualization.plot_distance1(self, n=n)



class TwoStepTomography2(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path

        # setup logging
        self.debug = debug
        self.setup_logging('TST2 - '+name)

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
            self.d['f_N0']       = None
            self.d['f_sample']   = None
            self.d['f_estimate'] = None
            self.d['f_distance'] = None

            # initialize dictionary
            for key in d.keys():
                self.d[key] = d[key]

            notNone = [v is not None for v in d.values()]
            assert all(notNone), 'Not all necessary parameters were initialized.'

            # initialize other attributes
            self.x_N     = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), self.d['N_ticks'], dtype=np.int)
            self.povm    = general.povm[self.d['povm_name']](self.d['dim'])

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

            self._a     = tst._a
            self._a_err = tst._a_err
            self._A     = tst._A
            self._A_err = tst._A_err

            self.logger.info('Loading already existing estimation data!')

        # report loaded parameters
        self.parameter_report()


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f"There are no sample states to reconstruct from."

        for j in range(self.d['N_mean']):
            self.logger.info(f"{j} of {self.d['N_mean']} states reconstructed.")
            for i in range(self.d['N_ticks']):
                self._estimates[j,i] = self.d['f_estimate'](self._originals[j], self.povm, self.x_N[i], self.d['f_N0'](self.x_N[i], self.d['alpha']), cup=self.d['cup'], mirror=self.d['mirror'])
                self._distances[j,i] = self.d['f_distance'](self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])
        self.logger.info(f"score of valid states: {np.sum(self._valids)/(self.d['N_mean']*self.d['N_ticks'])}")
        self.calculate_fitparam()


    def calculate_fitparam(self, n=0):
        assert n<self.d['N_ticks'], 'Parameter n is bigger than allowed!'

        f = lambda x, a, A: A * x**a

        mean = np.mean(self._distances, axis=0, where=self._valids)
        std  = np.std(self._distances, axis=0, where=self._valids)

        try:
            popt, pcov = curve_fit(f, self.x_N[n:], mean[n:])
            popt_err = np.sqrt(np.diag(pcov))

            self._a     = popt[0]
            self._A     = popt[1]
            self._a_err = np.sqrt(popt_err[0])
            self._A_err = np.sqrt(popt_err[1])
        except Exception as e:
            self.logger.info("Extracting parameters of overall scaling wasn't successful")
            self.logger.debug('The following error occurred in calculate_fitparam: '+str(e))

        return f, popt, popt_err


    def plot_distance(self, n=0):
        visualization.plot_distance2(self, n=n)


class TwoStepComparison(Comparison):

    def __init__(self, name, path, debug, name_list):

        self.name = name
        self.path = path

        # logging
        self.debug = debug
        self.setup_logging('TSC - '+self.name)

        # load data into
        self._list = []
        for name in name_list:
            try:
                tst1 = TwoStepTomography1(name, self.path, False, self.debug)
                tst1.d['N0']
                self._list.append(tst1)
                self.scheme = 1
            except:
                self._list.append(TwoStepTomography2(name, self.path, False, self.debug))
                self.scheme = 2

            self.logger.info(self._list[-1].parameter_report())

        tst_ref = self._list[0]

        # check comparison
        assert all([tst_ref.d['dim'] == tst.d['dim'] for tst in self._list]), 'Different dimension encountered. Comparison does not make sense!'
        assert all([tst_ref.d['N_max'] == tst.d['N_max'] for tst in self._list]), 'Different N_max encountered. Comparison does not make sense!'
        # assert all([tst_ref.d['f_sample'] == tst.d['f_sample'] for tst in self._list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([tst_ref.d['f_distance'] == tst.d['f_distance'] for tst in self._list]), 'Different distance measures encountered. Comparison does not make sense!'

        self.d = {}
        self.d['dim']        = tst_ref.d['dim']
        self.d['N_mean']     = ''
        self.d['N_max']      = tst_ref.d['N_max']
        self.d['f_sample']   = tst_ref.d['f_sample']
        self.d['f_distance'] = tst_ref.d['f_distance']


    def get_alpha(self):
        return [tst.d['alpha'] for tst in self._list]

    def get_mirror(self):
        return [tst.d['mirror'] for tst in self._list]

    def get_cup(self):
        return [tst.d['cup'] for tst in self._list]

    def get_sample(self):
        return [tst.d['f_sample'] for tst in self._list]


    def transform_citeria(self, criteria):

        data = {}
        data['f_estimate'] = [visualization.w[f_estimate] for f_estimate in self.get_estimation_method()]
        data['povm_name']  = self.get_povm_name()
        data['alpha']      = [fr"$\alpha$ = {alpha}" for alpha in self.get_alpha()]
        data['mirror']     = ['aligned' if mirror else 'anti-aligned' for mirror in self.get_mirror()]
        data['cup']        = ["$D_0\cup D_1$" if cup else "$D_1$" for cup in self.get_cup()]
        data['f_sample']   = [visualization.w[f_sample] for f_sample in self.get_sample()]

        return data[criteria]


    def compare_distance(self, criteria_1, criteria_2):
        if self.scheme==1:
            visualization.compare_distance(self, criteria_1, criteria_2)
        elif self.scheme==2:
            visualization.compare_distance_osc(self, criteria_1, criteria_2)



class TwoStepAlpha1(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name = name
        self.path = path

        # setup logging
        self.debug = debug
        self.setup_logging('TSA1 - '+self.name)

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
            self.povm    = general.povm[self.d['povm_name']](self.d['dim'])
            self.x_alpha = np.linspace(self.d['alpha_min'], self.d['alpha_max'], self.d['alpha_ticks'], endpoint=False, dtype=np.float)

            # initialize list where model will be stored in
            self._list = []

            self._a     = np.zeros((self.d['alpha_ticks'], 3), dtype=np.float)
            self._a_err = np.zeros((self.d['alpha_ticks'], 3), dtype=np.float)
            self._A     = np.zeros((self.d['alpha_ticks'], 3), dtype=np.float)
            self._A_err = np.zeros((self.d['alpha_ticks'], 3), dtype=np.float)
        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                tst = pickle.load(file)

            self.d       = tst.d
            self.x_alpha = tst.x_alpha
            self.povm    = tst.povm

            self._list = tst._list

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

            self._list.append(TwoStepTomography1(name, self.path, True, self.debug, tst_d))
            self._list[-1].create_originals()
            self._list[-1].reconstruct()

            self._a[idx], self._A[idx], self._a_err[idx], self._A_err[idx] = self._list[-1].get_scaling()
            self.logger.info(f"Tomography model for alpha = {alpha} initialized.")


    def review_fit(self, idx, n=0):
        self._list[idx].calculate_fitparam(n=n)
        self._list[idx].plot_distance(n=n)
        self._a[idx], self._A[idx], self._a_err[idx], self._A_err[idx] = self._list[idx].get_scaling()

    def plot_a_dependency(self, error=False):
        visualization.plot_a_dependency1(self, error=error)

    def plot_A_dependency(self, error=False):
        visualization.plot_A_dependency1(self, error=error)



class TwoStepAlpha2(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name = name
        self.path = path

        # setup logging
        self.debug = debug
        self.setup_logging('TSA2 - '+self.name)

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
            self.povm    = general.povm[self.d['povm_name']](self.d['dim'])
            self.x_alpha = np.linspace(self.d['alpha_min'], self.d['alpha_max'], self.d['alpha_ticks'], endpoint=False, dtype=np.float)

            # initialize list where model will be stored in
            self._list = []

            self._a     = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._a_err = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._A     = np.zeros(self.d['alpha_ticks'], dtype=np.float)
            self._A_err = np.zeros(self.d['alpha_ticks'], dtype=np.float)
        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                tst = pickle.load(file)

            self.d       = tst.d
            self.povm    = tst.povm
            self.x_alpha = tst.x_alpha

            self._list = tst._list

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

            self._list.append(TwoStepTomography2(name, self.path, True, self.debug, tst_d))
            self._list[-1].create_originals()
            self._list[-1].reconstruct()

            self._a[idx], self._A[idx], self._a_err[idx], self._A_err[idx] = self._list[-1].get_scaling()
            self.logger.info(f"Tomography model for alpha = {alpha} initialized.")


    def review_fit(self, idx, n=0):
        self._list[idx].calculate_fitparam(n=n)
        self._list[idx].plot_distance(n=n)
        self._a[idx], self._A[idx], self._a_err[idx], self._A_err[idx] = self._list[idx].get_scaling()

    def plot_a_dependency(self, error=False):
        visualization.plot_a_dependency2(self, error=error)

    def plot_A_dependency(self, error=False):
        visualization.plot_A_dependency2(self, error=error)
