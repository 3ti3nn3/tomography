import numpy as np
import logging
import pickle
import general
import const
import check
import simulate
import visualization


class Tomography:

    name  = 'base'
    new   = True
    debug = False

    dim     = None
    N       = None
    N_mean  = None
    x_N     = None

    f_sample   = None
    f_estimate = None
    f_distance = None

    _originals = None
    _esimates  = None
    _valids    = None
    _distances = None


    def setup_logging(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        handler     = logging.FileHandler('logs/'+self.name+'.log', mode='w')
        formatter   = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Tomography logger initialized.')

    def parameter_report(self):
        pass

    def get_originals(self):
        return self._originals

    def get_estimates(self):
        return self._estimates

    def get_distances(self):
        return self._distances

    def get_valids(self):
        return self._valids

    def plot_validity(self):
        visualization.plot_validity(self)

    def create_originals(self):
        self.logger.info('New set of original states will be created.')
        assert self._originals is None, f'Want to create new set of samples even though a set already exists.'

        self._originals = self.f_sample(self.dim, self.N_mean)

    def reconstruct(self):
        pass

    def plot_distance(self):
        visualization.plot_distance(self)

    def dispatch_model(self):
        with open('data/'+self.name+'.pt', "wb") as file:
            pickle.dump(self, file)
        self.logger.info('Dispatched model successfully.')


class Comparison:

    name  = 'base'
    new   = True
    debug = False

    N      = None
    N_mean = None

    f_sample   = None
    f_distance = None

    tomo_list = []

    def __init__(self):
        pass

    def setup_logging(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        handler     = logging.FileHandler('logs/'+self.name+'.log', mode='w')
        formatter   = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Tomography logger initialized.')


    def get_dim(self):
        return [tomo.dim for tomo in self.tomo_list]

    def get_estimation_method(self):
        return [tomo.f_estimate for tomo in self.tomo_list]

    def get_povm_name(self):
        return [tomo.povm_name for tomo in self.tomo_list]

    def compare_distance(self, criteria_1, criteria_2):
        visualization.compare_distance(self, criteria_1, criteria_2)

    def dispatch_model(self):
        with open('data/'+self.name+'.pt', "wb") as file:
            pickle.dump(self, file)
        self.logger.info('Dispatched model successfully.')
