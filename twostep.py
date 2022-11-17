from tomography import Tomography
from tomography import Comparison
import numpy as np
import pickle

import general
import check
import const
import simulate
import visualization


class TwoStepTomography(Tomography):


    def __init__(self, name, dim=None, N_max=None, N_mean=None, N_ticks=None, alpha=None, mirror=None, povm_name=None, f_sample=None, f_estimate=None, f_distance=None, new=True, debug=False):

        self.name  = name
        self.new   = new
        self.debug = debug

        # setup logging
        self.setup_logging('TST - '+self.name)

        # reload data
        if self.new:
            notNone = [dim is not None, N_max is not None, N_mean is not None, N_ticks is not None, alpha is not None, mirror is not None,\
                         povm_name is not None, f_sample is not None, f_estimate is not None, f_distance is not None]
            assert all(notNone), 'Want to build up model from scratch but certain variables are not specified.'

            self.logger.info('Buildung from scratch.')

            self.dim     = dim
            self.N_max   = N_max
            self.N_mean  = N_mean
            self.N_ticks = N_ticks
            self.x_N     = np.logspace(2, np.log10(self.N_max), self.N_ticks, dtype=np.int)
            self.alpha   = alpha
            self.mirror  = mirror

            self.povm_name  = povm_name
            self.povm       = const.povm[self.povm_name]
            self.f_sample   = f_sample
            self.f_estimate = f_estimate
            self.f_distance = f_distance

            self._originals = None
            self._estimates = np.empty((self.N_mean, self.N_ticks, self.dim, self.dim), dtype=np.complex)
            self._valids    = np.ones((self.N_mean, self.N_ticks), dtype=bool)
            self._distances = np.empty((self.N_mean, self.N_ticks), dtype=np.float)

        else:
            with open('data/'+self.name+'.pt', 'rb') as file:
                self.logger.info('Loading already existing estimation data!')
                tst = pickle.load(file)

            self.dim     = tst.dim
            self.N_max   = tst.N_max
            self.N_mean  = tst.N_mean
            self.N_ticks = tst.N_ticks
            self.x_N     = np.logspace(2, np.log10(self.N_max), self.N_ticks, dtype=np.int)
            self.alpha   = tst.alpha
            self.mirror  = tst.mirror

            self.povm_name  = tst.povm_name
            self.povm       = tst.povm
            self.f_sample   = tst.f_sample
            self.f_estimate = tst.f_estimate
            self.f_distance = tst.f_estimate

            self._originals = tst._originals
            self._estimates = tst._estimates
            self._valids    = tst._valids
            self._distances = tst._distances

        # report loaded parameters
        self.parameter_report()


    def parameter_report(self):
        info = '\n'\
            'Parameter report\n'\
            '----------------\n'\
            f'N_max            : {self.N_max}\n'\
            f'N_mean           : {self.N_mean}\n'\
            f'N_ticks          : {self.N_ticks}\n'\
            f'mirror           : {self.mirror}\n'\
            f'alpha            : {self.alpha}\n'\
            f'povm             : {self.povm_name}\n'\
            f'sample method    : {self.f_sample.__name__}\n'\
            f'distance measure : {self.f_distance.__name__}\n'\
            f'estimation method: {self.f_estimate.__name__}'
        self.logger.info(info)
        return info


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f'There are no sample states to reconstruct from.'

        for j in range(self.N_mean):
            for i in range(self.N_ticks):
                self._estimates[j,i] = self.f_estimate(self._originals[j], self.povm, self.x_N[i], mirror=self.mirror, alpha=self.alpha)
                self._distances[j,i] = self.f_distance(self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])

                if self.debug and not self._valids[j,i]:
                    self.logger.debug(f'\n'
                        'Error report of\n'\
                        '------------\n'\
                        f'eigenvalues, trace, hermitian: {general.state(self._estimates[j,i])}')

        self.logger.info(f'score of valid states: {np.sum(self._valids)/(self.N_mean*self.N_ticks)}')



class TwoStepComparison(Comparison):

    def __init__(self, name, name_list, new=True, debug=False):

        self.name = name
        self.new  = new

        # logging
        self.debug = debug
        self.setup_logging('TwoStepComparison')

        # load data
        for model_name in name_list:
            with open('data/'+model_name+'.pt', 'rb') as file:
                self.tomo_list.append(pickle.load(file))
                self.logger.info(self.tomo_list[-1].parameter_report())

        tst_ref = self.tomo_list[0]

        # check comparison
        assert all([tst_ref.N_mean == tst.N_mean for tst in self.tomo_list]), 'Different N_mean encountered. Comparison does not make sense!'
        assert all([tst_ref.N_max == tst.N_max for tst in self.tomo_list]), 'Different N_max encountered. Comparison does not make sense!'
        assert all([tst_ref.f_sample == tst.f_sample for tst in self.tomo_list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([tst_ref.f_distance == tst.f_distance for tst in self.tomo_list]), 'Different distance measures encountered. Comparison does not make sense!'

        self.N_mean     = tst_ref.N_mean
        self.N_max      = tst_ref.N_max
        self.f_sample   = tst_ref.f_sample
        self.f_distance = tst_ref.f_distance


    def get_alpha(self):
        return [tst.alpha for tst in self.tomo_list]

    def get_mirror(self):
        l = []
        for tst in self.tomo_list:
            if tst.mirror:
                l.append('anti-aligned')
            else:
                l.append('aligned')
        return l



class AlphaDependency(Tomography):

    def __init__(self, name, dim=None, N_max=None, N_mean=None, N_ticks=None, alpha_min=None, alpha_max=None, alpha_ticks=None, mirror=None, povm_name=None, f_sample=None, f_estimate=None, f_distance=None, new=True, debug=False):

        self.name  = name
        self.new   = new

        # setup logging
        self.debug = debug
        self.setup_logging('TST - '+self.name)

        # reload data
        if self.new:
            notNone = [dim is not None, N_max is not None, N_mean is not None, N_ticks is not None, alpha_min is not None,\
                alpha_max is not None, alpha_ticks is not None, mirror is not None, povm_name is not None,\
                f_sample is not None, f_estimate is not None, f_distance is not None]
            assert all(notNone), 'Want to build up model from scratch but certain variables are not specified.'

            self.logger.info('Buildung from scratch.')

            self.dim         = dim
            self.N_max       = N_max
            self.N_mean      = N_mean
            self.N_ticks     = N_ticks
            self.x_N         = np.logspace(2, np.log10(self.N_max), self.N_ticks, dtype=np.int)
            self.alpha_min   = alpha_min
            self.alpha_max   = alpha_max
            self.alpha_ticks = alpha_ticks
            self.mirror      = mirror

            self.povm_name  = povm_name
            self.povm       = const.povm[self.povm_name]
            self.f_sample   = f_sample
            self.f_estimate = f_estimate
            self.f_distance = f_distance

            self._alpha_list = []
            self._a          = np.zeros(self.alpha_ticks, dtype=np.float)
            self._a_err      = np.zeros(self.alpha_ticks, dtype=np.float)
            self._A          = np.zeros(self.alpha_ticks, dtype=np.float)
            self._A_err      = np.zeros(self.alpha_ticks, dtype=np.float)
        else:
            with open('data/'+self.name+'.pt', 'rb') as file:
                self.logger.info('Loading already existing estimation data!')
                model = pickle.load(file)

            self.dim         = model.dim
            self.N_max       = model.N_max
            self.N_mean      = model.N_mean
            self.N_ticks     = model.N_ticks
            self.x_N         = np.logspace(2, np.log10(self.N_max), self.N_ticks, dtype=np.int)
            self.alpha_min   = model.alpha_min
            self.alpha_max   = model.alpha_max
            self.alpha_ticks = model.alpha_ticks
            self.mirror      = model.mirror

            self.povm_name  = model.povm_name
            self.povm       = model.povm
            self.f_sample   = model.f_sample
            self.f_estimate = model.f_estimate
            self.f_distance = model.f_estimate

            self._alpha_list = model._alpha_list
            self._a          = model._a
            self._a_err      = model._a_err
            self._A          = model._A
            self._A_err      = model._A_err

        # report loaded parameters
        self.parameter_report()


    def parameter_report(self):
        info = '\n'\
            'Parameter report\n'\
            '----------------\n'\
            f'N_max            : {self.N_max}\n'\
            f'N_mean           : {self.N_mean}\n'\
            f'N_ticks          : {self.N_ticks}\n'\
            f'mirror           : {self.mirror}\n'\
            f'alpha_min        : {self.alpha_min}\n'\
            f'alpha_max        : {self.alpha_max}\n'\
            f'alpha_ticks      : {self.alpha_ticks}\n'\
            f'povm             : {self.povm_name}\n'\
            f'sample method    : {self.f_sample.__name__}\n'\
            f'distance measure : {self.f_distance.__name__}\n'\
            f'estimation method: {self.f_estimate.__name__}'
        self.logger.info(info)
        return info


    def get_alpha_list(self):
        return self._alpha_list

    def get_a(self):
        return self._a

    def get_A(self):
        return self._A


    def create_models(self):
        for alpha in self.x_alpha:
            name = self.name+'_'+str(alpha)
            self._alpha_list.append(TwoStepTomography(name, self.dim, self.N_max,\
                self.N_mean, self.N_ticks, alpha, self.mirror, self.povm_name, self.f_sample,\
                self.f_estimate, self.f_distance, new=True, debug=self.debug))

            self._alpha_list[-1].create_originals()
            self._alpha_list[-1].reconstruct()

            self.logger.info(f'Tomography model for alpha = {alpha} initialized.')


    def extract_gradient(self):
        for tomo, idx in enumerate(self._alpha_list):

            # calculate mean
            mean = np.mean(tomo.get_distances(), axis=0, where=tomo.get_valids())
            std  = np.std(tomo.get_distances(), axis=0, where=tomo.get_valids())

            # fit
            try:
                f = lambda x, a, A: A*x**a
                popt, pcov = curve_fit(f, tomo.x_N, mean, p0=[-0.5, 0.2], sigma=std)

                self._a[idx], self._A[idx] = popt
                self._a_err = np.sqrt(pcov[0,0])
                self._A_err = np.sqrt(popt[0,0])

                self.logger.info('curve_fit successful!')
            except:
                 self.logger.info('curve_fit not successful!')


    def plot_alpha_dependency(self):
        pass
