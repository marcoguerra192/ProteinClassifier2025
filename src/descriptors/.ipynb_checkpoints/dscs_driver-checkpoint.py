''' 
/* SHREC 2025
Marco Guerra

*/
'''

# Descriptors for data

import sys, os

from pathlib import Path

import numpy as np
from gudhi import AlphaComplex
from gudhi.representations import ProminentPoints, PersistenceImage

from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK
from .alpha_prominent import AlphaDiag, PersImagesVectorize
from .distance_dist import quantiles_of_distance, centroid

import csv

def compute_descriptors( data_source : str | Path, model : str, **kwargs ):
    '''Helper function to compute descriptors choosing the appropriate 
    combination of methods from below

    '''

    ## parse optional arguments
    Num_Prominent = kwargs.get('Num_Prominent', None)  # Default to None if not provided
    PersImPoints = kwargs.get('PersImPoints', None)
    which_quantiles = kwargs.get('which_quantiles', None)

    source = DataSource( data_source, base_path=data_source)
    N_Files = len(list(source))
    
    # vector of labels
    labels = []

    if model == 'AlphaProminent':
        data = np.zeros( (N_Files , 3*(PersImPoints**2)) )

    if model == 'quantiles':
        data = np.zeros( ( N_Files , len(which_quantiles)) )

    if model == 'sublevel_topology':
        data = np.zeros((0,0))
    
    # read the labels
    Truth = {}
    with open('./data/data/train_set.csv', 'r') as csvfile:
        truths = csv.reader(csvfile)
    
        next(truths, None) # skip first row, it's a header
        for t in truths:
            Truth[t[0]] = int(t[1])
    
    source = DataSource( data_source, base_path=data_source)

    for j,s in enumerate(source):

        print(j+1, 'out of', N_Files)
        print(s)

        # Find the label for the protein we are reading
        filename = os.path.basename(s)
        filename = os.path.splitext(filename)[0]
        labels.append(Truth[filename])


        if model == 'AlphaProminent':

            points = read_vertices_VTK(s , out_var=True).tolist()
            
            Dgm0, Dgm1, Dgm2 = AlphaDiag(points, N_Prominent = Num_Prominent)
            Img = PersImagesVectorize(Dgm0, Dgm1, Dgm2, res = PersImPoints)
            data[j,:] = Img

        if model == 'quantiles':

            points = read_vertices_VTK(s , out_var=True)

            center = centroid(points)

            quantiles = quantiles_of_distance(points, center, which_quantiles)
            data[j,:] = np.array(quantiles).reshape( (1, len(which_quantiles)) )

        if model == 'sublevel_topology':

            print('Implement!')

            

            
            

    return data, labels

















