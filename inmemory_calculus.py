from __future__ import division

import os

import numpy as np
import h5py
from pathlib import Path

inmem = {}

def __load_inmemory_calculus(path, key, dataset_key=u'Dataset1'):
    def read_h5_data(data_path):
        data_sets = {}

        with h5py.File(data_path, 'r') as py_file:
            for key in py_file.keys():
                data_sets[key] = py_file[key][:]

        return data_sets

    inmem[key] = read_h5_data(path)[dataset_key]

def load_inmemory_calculus(type):
    root_path = Path(os.path.abspath('inmemory_calculus'))

    __load_inmemory_calculus(root_path.joinpath(type).joinpath('R.h5'), 'REACTIVE_UNIT_DIST')
    __load_inmemory_calculus(root_path.joinpath(type).joinpath('RxR.h5'), 'REACTIVE_X_REACTIVE')
    __load_inmemory_calculus(root_path.joinpath(type).joinpath('domain.h5'), 'DOMAIN')
    if type == 'quotient': # for quotient based we expect the file with reducted quotients
        __load_inmemory_calculus(root_path.joinpath(type).joinpath('nklist.h5'), 'STIMULUS_LIST')
    if type == 'numeric':
        inmem['STIMULUS_LIST'] = np.arange(1, len(inmem['REACTIVE_UNIT_DIST']) + 1)

    STIMULUS_LIST = inmem['STIMULUS_LIST']
    REACTIVE_UNIT_DIST = inmem['REACTIVE_UNIT_DIST']
    REACTIVE_X_REACTIVE = inmem['REACTIVE_X_REACTIVE']
    DOMAIN = inmem['DOMAIN']

    # VALIDATE loaded data types:
    if not isinstance(STIMULUS_LIST, np.ndarray):
        raise ValueError('Expected nk_list to be numpy array type, found {} type'.format(type(STIMULUS_LIST)))
    if not isinstance(REACTIVE_UNIT_DIST, np.ndarray):
        raise ValueError('Expected ? to be numpy array, found {} type'.format(type(REACTIVE_UNIT_DIST)))
    if not isinstance(REACTIVE_X_REACTIVE, np.ndarray):
        raise ValueError('Expected ? to be numpy array, found {} type'.format(type(REACTIVE_X_REACTIVE)))
    if not isinstance(DOMAIN, np.ndarray):
        raise ValueError('Expected ? to be numpy array, found {} type'.format(type(DOMAIN)))

    # VALIDATE loaded data shapes:
    if not STIMULUS_LIST.shape[0] == REACTIVE_UNIT_DIST.shape[0] == REACTIVE_X_REACTIVE.shape[0] == REACTIVE_X_REACTIVE.shape[1]:
        raise ValueError()
    if not REACTIVE_UNIT_DIST.shape[1] == DOMAIN.shape[0]:
        raise ValueError()
