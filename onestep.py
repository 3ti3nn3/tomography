from tomography import Tomography
from tomography import Comparison
import numpy as np
import pickle
import general
import check
import const
import simulate
import visualization
from scipy.optimize import curve_fit


class OneStepTomography(Tomography):

    def __init__(self, name, path, new, debug, d=None):

        self.name  = name
        self.path  = path
        self.new   = new

        # setup logging
        self.debug = debug
        self.setup_logging('OST - '+self.name)

        # reload data
        if self.new:
            assert d is not None, 'Want to build up model from scratch but certain variables are not specified.'
            self.logger.info('Buildung from scratch.')

            # add new parameters
            self.d = {}
            self.d['dim']       = None
            self.d['N_min']     = None
            self.d['N_max']     = None
            self.d['N_ticks']   = None
            self.d['N_mean']    = None
            self.d['povm_name'] = None
            self.d['f_sample']   = None
            self.d['f_estimate'] = None
            self.d['f_distance'] = None

            # initialize dictionary
            for key in d.keys():
                self.d[key] = d[key]

            notNone = [v is not None for v in d.values()]
            assert all(notNone), 'Not all necessary parameters were initialized.'

            # initialize other attributes
            self.x_N  = np.logspace(np.log10(self.d['N_min']), np.log10(self.d['N_max']), self.d['N_ticks'], dtype=np.int)
            self.povm = general.povm[self.d['povm_name']]

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
                self.logger.info('Loading already existing estimation data!')
                ost = pickle.load(file)

                self.d    = ost.d
                self.x_N  = ost.x_N
                self.povm = ost.povm

                self._originals = ost._originals
                self._estimates = ost._estimates
                self._valids    = ost._valids
                self._distances = ost._distances

                self._a     = ost._a
                self._a_err = ost._a_err
                self._A     = ost._A
                self._A_err = ost._A_err

        # report loaded parameters
        self.parameter_report()


    def reconstruct(self):
        self.logger.info('New estimates will be constructed.')
        assert self._originals is not None, f"There are no sample states to reconstruct from."

        for j in range(self.d['N_mean']):
            self.logger.info(f"{j} of {self.d['N_mean']} states have been reconstructed so far!")

            measurement = simulate.measure(self._originals[j], self.d['N_max'], self.povm)
            for i in range(self.d['N_ticks']):
                self._estimates[j,i] = self.d['f_estimate'](measurement[:self.x_N[i]], self.povm)
                self._distances[j,i] = self.d['f_distance'](self._originals[j], self._estimates[j,i])
                self._valids[j,i]    = check.state(self._estimates[j,i])

        self.logger.info(f"score of valid states: {np.sum(self._valids)/(self.d['N_mean']*self.d['N_ticks'])}")


    def calculate_fitparam(self, n=0):

        f = lambda x, a, A: A * x**a

        mean = np.mean(self._distances, axis=0, where=self._valids)
        std  = np.std(self._distances, axis=0, where=self._valids)

        try:
            popt, pcov = curve_fit(f, self.x_N[n:], mean[n:], sigma=std[n:])
            popt_err = np.sqrt(np.diag(pcov))

            self._a     = popt[0]
            self._A     = popt[1]
            self._a_err = popt_err[0]
            self._A_err = popt_err[1]
        except Exception as e:
            self.logger.info("Extracting parameters of overall scaling wasn't successful")
            self.logger.debug('The following error occurred in calculate_fitparam: '+str(e))
            popt     = [None, None]
            popt_err = [None, None]

        return f, popt, popt_err


    def plot_distance(self):
        visualization.plot_distance2(self)


class OneStepComparison(Comparison):

    def __init__(self, name, path, debug, name_list):

        self.name = name
        self.path = path

        # logging
        self.debug = debug
        self.setup_logging('OSC - '+self.name)

        # load data
        self._list = []
        for model_name in name_list:
            with open(self.path+'data/'+model_name+'.pt', 'rb') as file:
                self._list.append(pickle.load(file))

        ost_ref = self._list[0]

        # check comparison
        assert all([ost_ref.d['N_mean'] == ost.d['N_mean'] for ost in self._list]), 'Different N_mean encountered. Comparison does not make sense!'
        assert all([ost_ref.d['N_max'] == ost.d['N_max'] for ost in self._list]), 'Different N_max encountered. Comparison does not make sense!'
        assert all([ost_ref.d['f_sample'] == ost.d['f_sample'] for ost in self._list]), 'Different way of sampling encountered. Comparison does not make sense!'
        assert all([ost_ref.d['f_distance'] == ost.d['f_distance'] for ost in self._list]), 'Different distance measures encountered. Comparison does not make sense!'

        self.d = {}
        self.d['N_min']   = ost_ref.d['N_min']
        self.d['N_max']   = ost_ref.d['N_max']
        self.d['N_ticks'] = ost_ref.d['N_ticks']

        self.d['N_mean']     = ost_ref.d['N_mean']
        self.d['f_sample']   = ost_ref.d['f_sample']
        self.d['f_distance'] = ost_ref.d['f_distance']


    def get_povm_name(self):
        return [ost.d['povm_name'] for ost in self._list]

    def transform_citeria(self, criteria):

        data = {}
        data['f_estimate'] = [visualization.w[f_estimate] for f_estimate in self.get_estimation_method()]
        data['povm_name']  = self.get_povm_name()

        return data[criteria]
