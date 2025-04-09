import sys,os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK
from src.descriptors.dscs_driver import compute_descriptors

from gudhi.representations import ProminentPoints
from gudhi.representations import PersistenceImage


import csv

# number of prominent points to keep for each dimension for each PD
Num_Prominent = 150
# resolution of each axis for the persistent images
PersImPoints = 5

# data base path
data_base = './data/data/'
# data source
data_source = os.path.join(data_base, 'train_set/')

# set params for the run
#Model = 'AlphaProminent'
#Model = 'quantiles'
#Model = 'Combined'
Model = 'Sectors'

params = { 'data':data_base}
params.update(  { 'Num_Prominent' : Num_Prominent , 'PersImPoints' : PersImPoints } )
params.update( { 'which_quantiles':[ .25, .5 , .75 ] } )

data, labels = compute_descriptors(data_source, model=Model, **params)


# save descriptors
save_descs_path = os.path.join(data_base,'saved_descriptors/train_set/')

# save
np.save(os.path.join(save_descs_path, Model+'Data.npy'), data)
np.save(os.path.join(save_descs_path, Model+'Labels.npy'), np.array(labels) )
