from tomography import Tomography
from tomography import Comparison
import numpy as np
import pickle
import general
import check
import const
import simulate
import visualization


class OneStepTomography(Tomography):

    def __init__(self, name, dim=None, N=None, N_mean=None, povm_name=None, f_sample=None, f_estimate=None, f_distance=None, new=True, debug=False):

        self.name  = name
        self.new   = new
        self.debug = debug

        # setup logging
        self.setup_logging('OneStepTomography')

        # reload data
        if self.new:
            notNone = [dim is not None, N is not None, N_mean is not None,\
                         povm_name is not None, f_sample is not None, f_estimate is not None, f_distance is not None]
            assert all(notNone), 'Want to build up model from scratch but certain variables are not specified.'
            assert len(N)==3, f'Unexpexted shape of N encountered: {N}'

            self.logger.info('Buildung from scratch.')

            self.dim     = dim
            self.N       = N
            self.N_mean  = N_mean
            self.x_N     = np.logspace(np.log10(self.N[0]), np.log10(self.N[1]), self.N[2], dtype=np.int)

            self.povm_name  = povm_name
            self.povm       = const.povm[self.povm_name]
            self.f_sample   = f_sample
            self.f_estimate = f_estimate
            self.f_distance = f_distance

            self._originals = None
            self._estimates = np.empty((self.N_mean, self.N[2], self.dim, self.dim), dtype=np.complex)
            self._valids    = np.ones((self.N_mean, self.N[2]), dtype=bool)
            self._distances = np.empty((self.N_mean, self.N[2]), dtype=np.float)

        else:
            with open('data/'+self.name+'.pt', 'rb') as file:
                self.logger.info('Loading already existing estimation data!')
                ost = pickle.load(file)

                try:
                    self.N = ost.N
                except:
                    self.N = [int(1e02), ost.N_max, ost.N_ticks]

                self.dim     = ost.dim
                self.N_mean  = ost.N_mean
                self.x_N     = np.logspace(np.log10(self.N[0]), np.log10(self.N[1]), self.N[2], dtype=np.int)

                self.povm_name  = ost.povm_name
                self.povm       = ost.povm
                self.f_sample   = ost.f_sample
                self.f_estimate = ost.f_estimate
                self.f_distance = ost.f_estimate

                self._originals = ost._originals
                self._estimates = ost._estimates
                self._valids    = ost._valids
                self._distances = ost._distances

        # report loaded parameters
        self.parameter_report()


    def parameter_report(self):
        info = '\n'\
            'Parameter report\n'\
            '----------------\n'\
            f'N                : {self.N}\n'\
            f'N_mean           : {self.N_mean}\n'\
            f'povm             : {self.povm_name}\n'\
            f'sample method    : {self.f_sample.__name__}\n'\
            f'distance measure : {self.f_distance.__name__}\n'\
            f'estimation method: {self.f_estimate.__module__}.{self.f_estimate.__name__}'
        self.logger.info(info)
        return info


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f'There are no sample states to reconstruct from.'

        for j in range(self.N_mean):
            measurement = simulate.measure(self._originals[j], self.N[1], self.povm)
            for i in range(self.N[2]):
                self._estimates[j,i] = self.f_estimate(measurement[:self.x_N[i]], self.povm)
                self._distances[j,i] = self.f_distance(self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])

                if self.debug and not self._valids[j,i]:
                    self.logger.debug(f'\n'
                        'Error report\n'\
                        '------------\n'\
                        f'eigenvalues, trace, hermitian: {general.state(self._estimates[j,i])}')

        self.logger.info(f'score of valid states: {np.sum(self._valids)/(self.N_mean*self.N[2])}')


class OneStepComparison(Comparison):

    def __init__(self, name, name_list, debug=False):

        self.name = name

        # logging
        self.debug = debug
        self.setup_logging()

        # load data
        for model_name in name_list:
            with open('data/'+model_name+'.pt', 'rb') as file:
                self.tomo_list.append(pickle.load(file))

        ost_ref = self.tomo_list[0]

        # check comparison
        assert all([ost_ref.N_mean == ost.N_mean for ost in self.tomo_list]), 'Different N_mean encountered. Comparison does not make sense!'
        assert all([ost_ref.N[1] == ost.N[1] for ost in self.tomo_list]), 'Different N_max encountered. Comparison does not make sense!'
        assert all([ost_ref.f_sample == ost.f_sample for ost in self.tomo_list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([ost_ref.f_distance == ost.f_distance for ost in self.tomo_list]), 'Different distance measures encountered. Comparison does not make sense!'

        try:
            self.N = ost_ref.N
        except:
            self.N = [int(1e02), ost_ref.N_max, ost_ref.N_ticks]

        self.N_mean     = ost_ref.N_mean
        self.f_sample   = ost_ref.f_sample
        self.f_distance = ost_ref.f_distance


    def get_povm_name(self):
        return [ost.povm_name for ost in self.tomo_list]
