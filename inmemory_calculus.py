from __future__ import division

import h5py


def __read_h5_data(data_path):
    data_sets = {}

    with h5py.File(data_path, 'r') as py_file:
        for key in py_file.keys():
            data_sets[key] = py_file[key][:]

    return data_sets

nklist = __read_h5_data('C:\Users\kuba\Workspaces\coordinating-quantifiers\inmemory_calculus\\nklist.h5')
discrete_Ri = __read_h5_data('C:\Users\kuba\Workspaces\coordinating-quantifiers\inmemory_calculus\\discrete_Ri.h5')
elements = __read_h5_data('C:\Users\kuba\Workspaces\coordinating-quantifiers\inmemory_calculus\\elements.h5')
x = __read_h5_data('C:\Users\kuba\Workspaces\coordinating-quantifiers\inmemory_calculus\\x.h5')

NK_LIST = nklist[u'Dataset1']
DISCRETE_RI = discrete_Ri[u'Dataset1']
RXR = elements[u'Dataset1']
DOMAIN = x[u'Dataset1']

