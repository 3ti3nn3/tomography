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

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path
        self.new   = new

        # setup logging
        self.debug = debug
        self.setup_logging('OST - '+{self.name})

        # reload data
        if self.new:
            assert d is not None, 'Want to build up model from scratch but certain variables are not specified.'
            self.logger.info('Buildung from scratch.')

            self.d         = d
            self.x_N  = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), self.d['N_ticks'], dtype=np.int)
            self.povm = const.povm[self.d['povm_name']]

            self._originals = None
            self._estimates = np.empty((self.d['N_mean'], self.d['N_ticks'], slef.d['dime'], self.d['dim']), dtype=np.complex)
            self._valids    = np.ones((self.d['N_mean'], self.d['N_ticks']), dtype=bool)
            self._distances = np.empty((self.d['N_mean'], self.d['N_ticks']), dtype=np.float)

        else:
            with open(self.path+'data/'+self.name+'.pt', 'rb') as file:
                self.logger.info('Loading already existing estimation data!')
                ost = pickle.load(file)

                self.d = ost.d

                self._originals = ost._originals
                self._estimates = ost._estimates
                self._valids    = ost._valids
                self._distances = ost._distances

        # report loaded parameters
        self.parameter_report()


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f"There are no sample states to reconstruct from."

        for j in range(self.d['N_mean']):
            measurement = simulate.measure(self._originals[j], self.d['N_max'], self.povm)
            for i in range(self.d['N_max']):
                self._estimates[j,i] = self.d['f_estimate'](measurement[:self.x_N[i]], self.povm)
                self._distances[j,i] = self.d['f_distance'](self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])

                if self.debug and not self._valids[j,i]:
                    self.logger.debug(f"\n"
                        'Error report\n'\
                        '------------\n'\
                        f"eigenvalues, trace, hermitian: {general.state(self._estimates[j,i])}")

        self.logger.info(f"score of valid states: {np.sum(self._valids)/(self.d['N_mean']*self.d['N_ticks'])}")


class OneStepComparison(Comparison):

    def __init__(self, name, path, name_list, debug=False):

        self.name = name
        self.path = path

        # logging
        self.debug = debug
        self.setup_logging()

        # load data
        for model_name in name_list:
            with open(self.path+'data/'+model_name+'.pt', 'rb') as file:
                self._list.append(pickle.load(file))

        ost_ref = self._list[0]

        # check comparison
        assert all([ost_ref.d['N_mean'] == ost.d['N_mean'] for ost in self._list]), 'Different N_mean encountered. Comparison does not make sense!'
        assert all([ost_ref.d['N_max'] == ost.d['N_max'] for ost in self._list]), 'Different N_max encountered. Comparison does not make sense!'
        assert all([ost_ref.d['f_sample'] == ost.d['f_sample'] for ost in self._list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([ost_ref.d['f_distance'] == ost.d['f_distance'] for ost in self._list]), 'Different distance measures encountered. Comparison does not make sense!'

        self.d['N_min']   = ost_ref.d['N_min']
        self.d['N_max']   = ost_ref.d['N_min']
        self.d['N_ticks'] = ost_ref.d['N_ticks']

        self.d['N_mean']     = ost_ref.d['N_mean']
        self.d['f_sample']   = ost_ref.d['f_sample']
        self.d['f_distance'] = ost_ref.d['f_distance']


    def get_povm_name(self):
        return [ost.d['povm_name'] for ost in self._list]
