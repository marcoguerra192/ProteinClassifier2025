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
from .distance_dist import quantiles_of_distance, centroid, distances_from_point

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

    if model == 'Combined':

        # 3 quantiles, 3 sumulative radial potential, num of pairs with lifetime > threshold
        # for dim 0,1,2 , birth and death of longest H_0 bar, potential at corresponding 
        # generating vertices
        N_Features = 3 + 3 + 3 + 2 + 2
        
        data = np.zeros(( N_Files ,N_Features) , dtype = float)
    
    # read the labels
    Truth = {}
    with open('./data/data/train_set.csv', 'r') as csvfile:
        truths = csv.reader(csvfile)
    
        next(truths, None) # skip first row, it's a header
        for t in truths:
            Truth[t[0]] = int(t[1])
    
    source = DataSource( data_source, base_path=data_source)

    for j,s in enumerate(source):

        print(j+1, 'out of', N_Files, flush=True)
        print(s, flush=True)

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

        if model == 'Combined':

            
            row = np.zeros( ( N_Features ) , dtype = float)

            read_data_file = './data/data/train_set_Numpy/' + filename + '.vtk.npz'
            res = np.load( read_data_file, allow_pickle=False )

            points = res['points']
            center = centroid(points)

            N_Verts = points.shape[0]

            quantiles = quantiles_of_distance(points, center, which_quantiles)
            
            row[0:3] = np.array(quantiles).reshape( (1, len(which_quantiles)) )

            dists = distances_from_point(points, center) # vector of distances of each point from centroid

            ordering = np.argsort(dists) # sort them by distance
            
            potentials = res['potentials']

            radial_charge = np.cumsum( potentials[ordering] ) # potentials ordered by closest to farthest from centroid, cumulative sum

            significant_entries = [ np.floor( x * N_Verts ) for x in which_quantiles] # entries of radial charge corresponding to the desired quantiles
            significant_entries = np.array(significant_entries, dtype = int)

            cumulative_charge_at_quantiles = radial_charge[ significant_entries ] # pick the corresponding charges

            row[3:6] = cumulative_charge_at_quantiles

            read_pers_file = './data/data/sublevelset_filtrations/train_set/' + filename + '.vtk.npz'

            res = np.load( read_pers_file, allow_pickle=True )

            dgm0 = res['dgm0']
            dgm1 = res['dgm1']
            dgm2 = res['dgm2']
            gens = res['gens']

            Lambda = 1.0/10
            Threshold = quantiles[1] * Lambda # TRY ONE TENTH OF THE MEDIAN

            pers0 = dgm0[:,1] - dgm0[:,0]
            pers1 = dgm1[:,1] - dgm1[:,0]
            pers2 = dgm2[:,1] - dgm2[:,0]

            pers0 = dgm0[:,1] - dgm0[:,0]
            mask0 = (pers0 >= Threshold) & (pers0 < np.inf)
            count0 = np.sum(mask0)

            pers1 = dgm1[:,1] - dgm1[:,0]
            mask1 = (pers1 >= Threshold) & (pers1 < np.inf)
            count1 = np.sum(mask1)

            pers2 = dgm2[:,1] - dgm2[:,0]
            mask2 = (pers2 >= Threshold) & (pers2 < np.inf)
            count2 = np.sum(mask2)

            row[6:9] = np.array([ count0 , count1 , count2 ])

            longest0_ind = np.argmax(pers0[ pers0 < np.inf])
            
            longest0 = dgm0[longest0_ind,:] # longest interval in H_0

            row[9:11] = longest0

            # Find generating vertices and their potentials
            gen_verts = gens[0][0][longest0_ind,:] # gens[0] is finite pairs, [0] is dimension 0, next index is which entry

            gen_potentials = potentials[ gen_verts ].reshape((2,)) # Find the potentials at the vertices creating the largest CC

            row[11:] = gen_potentials

            data[j,:] = row
            
            

    return data, labels

















