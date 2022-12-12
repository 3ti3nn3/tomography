import numpy as np
import logging
import pickle
import general
import const
import check
import simulate
import visualization
import shutil


class Tomography:

    '''
    self.name  = 'base'
    self.path  = None
    self.new   = True
    self.debug = False

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

    self.povm = None
    self.x_N  = None

    self._originals = None
    self._esimates  = None
    self._valids    = None
    self._distances = None

    self._a     = None
    self._a_err = None
    self._A     = None
    self._A_err = None
    '''

    def setup_logging(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        handler     = logging.FileHandler(self.path+'logs/'+self.name+'.log', mode='w')
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
        info = '\n'+f"Parameter report of {self.name}\n"+'------------------------------\n'
        for key in self.d:
            info += '{0:15}:{1}\n'.format(key, self.d[key])
        self.logger.info(info)
        return info

    def update_param(self, key, value):
        self.logger.info(f"Updating {key} parameter from {self.d[key]} to {value}.")
        self.d[key] = value

    def get_originals(self):
        return self._originals

    def get_estimates(self):
        return self._estimates

    def get_distances(self):
        return self._distances

    def get_valids(self):
        return self._valids

    def get_scaling(self):
        return self._a, self._A, self._a_err, self._A_err

    def create_originals(self):
        self.logger.info('New set of original states will be created.')
        assert self._originals is None, f"Want to create new set of samples even though a set already exists."

        self._originals = self.d['f_sample'](self.d['dim'], self.d['N_mean'])

    def reconstruct(self):
        pass

    def calculate_fitparam(self):
        pass

    def plot_distance(self):
        pass

    def plot_validity(self):
        visualization.plot_validity(self)

    def dispatch_model(self, path=''):
        with open(self.path+'data/'+self.name+'.pt', "wb") as file:
            pickle.dump(self, file)
        self.logger.info('Dispatched model successfully.')

        if path!='':
            shutil.move(self.path+'logs/'+self.name+'.log', path+'logs/'+self.name+'.log')
            shutil.move(self.path+'data/'+self.name+'.pt', path+'data/'+self.name+'.pt')
            try:
                shutil.move(self.path+'plots/val_'+self.name+'.png', path+'plots/val_'+self.name+'.png')
            except Exception as e:
                self.logger.debug(f"The following error occurred in dispatch_model: {e}")
            try:
                shutil.move(self.path+'plots/dist_'+self.name+'.png', path+'plots/dist_'+self.name+'.png')
            except Exception as e:
                self.logger.debug(f"The following error occurred in dispatch_model: {e}")
            try:
                shutil.move(self.path+'plots/alpha_'+self.name+'.png', path+'plots/alpha_'+self.name+'.png')
            except Exception as e:
                self.logger.debug(f"The following error occurred in dispatch_model: {e}")



class Comparison:

    '''
    self.name  = 'base'
    self.path  = None
    self.new   = True
    self.debug = False

    self.d = {}
    self.d['dim']    = None
    self.d['N_max']  = None
    self.d['N_mean'] = None
    self.d['f_sample']   = None
    self.d['f_distance'] = None

    self._list = None
    '''

    def setup_logging(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        handler     = logging.FileHandler(self.path+'logs/'+self.name+'.log', mode='w')
        formatter   = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Comparison logger initialized.')

    def parameter_report(self):
        info = '\n'+'Parameter report\n'+'----------------\n'
        for key in self.d:
            info += '{0:15}:{1}\n'.format(key, self.d[key])
        self.logger.info(info)
        return info

    def get_estimation_method(self):
        return [tomo.d['f_estimate'] for tomo in self._list]

    def get_povm_name(self):
        return [tomo.d['povm_name'] for tomo in self._list]

    def get_N_min(self):
        return [tomo.d['N_min'] for tomo in self._list]

    def transform_citeria(self, criteria):
        pass

    def compare_distance(self, criteria_1, criteria_2):
        visualization.compare_distance_osc(self, criteria_1, criteria_2)


    def dispatch_model(self, path=''):
        with open(self.path+'data/'+self.name+'.pt', "wb") as file:
            pickle.dump(self, file)
        self.logger.info('Dispatched model successfully.')

        if path!='':
            shutil.move(self.path+'logs/'+self.name+'.log', path+'logs/'+self.name+'.log')
            shutil.move(self.path+'data/'+self.name+'.log', path+'data/'+self.name+'.log')
            try:
                shutil.move(self.path+'plots/comp_'+self.name+'.png', path+'data/comp_'+self.name+'.png')
            except Exception as e:
                self.logger.debug(f"The following error occurred in dispatch_model: {e}")
