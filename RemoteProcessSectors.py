import sys,os
from pathlib import Path
import numpy as np

from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK, convertVTK_to_numpy
from src.descriptors.distance_dist import quantiles_of_distance, distances_from_point
from src.descriptors.distance_dist import centroid as Centroid

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from gudhi import SimplexTree
from gudhi import RipsComplex, AlphaComplex, plot_persistence_diagram

import joblib

import csv
import pickle

def which_sector( p , centroid ):
    '''Find which of the 8 sectors a point belongs to, wrt the centroid

    '''
    vect = p - centroid

    if vect[2] >= 0.0: # z >= 0.0
        if vect[0] >= 0.0: # x >= 0.0

            if vect[1] >= 0.0: # y >= 0.0
                return 1
            else: # y < 0.0
                return 4
        else: # x < 0.0
            if vect[1] >= 0.0: # y >= 0.0
                return 2
            else: # y < 0.0
                return 3
        
    else: # z < 0.0
        if vect[0] >= 0.0: # x >= 0.0

            if vect[1] >= 0.0: # y >= 0.0
                return 5
            else: # y < 0.0
                return 8
        else: # x < 0.0
            if vect[1] >= 0.0: # y >= 0.0
                return 6
            else: # y < 0.0
                return 7

def common_sector( simplex, centroid ):
    '''Find which common sector a list of vertices belongs to. Raise
    ValueError if it does not exist

    '''
    sectors = np.array([which_sector(p, centroid) for p in simplex], dtype = int)
    unique_sec = np.unique(sectors)

    if unique_sec.shape[0] > 1:
        raise ValueError

    else:
        return unique_sec[0]
        
    


source = DataSource( './data/data/train_set/', base_path='./data/data/train_set')
N_Files = len(list(source))

source = DataSource( './data/data/train_set/', base_path='./data/data/train_set/')

which_quantiles = [ 0.25 , 0.5 , 0.75 ]

for j,s in enumerate(source):

    print(j+1, 'out of', N_Files, flush=True)
    print(s, flush=True)

    filename = os.path.basename(s)
    filename, _ = os.path.splitext(filename)

    # read points, triangles, potentials, etc

    read_data_file = './data/data/train_set_Numpy/' + filename + '.vtk.npz'
    res = np.load( read_data_file, allow_pickle=False )

    points = res['points']
    triangles = res['triangles']
    potentials = res['potentials']

    N_Verts = points.shape[0]

    centroid = Centroid(points)
    dists = distances_from_point(points, centroid)

    ## NOW SPLIT IN 8 SECTORS

    SecPoints = []
    SCs = []
    for _ in range(8):
        SCs.append( SimplexTree() )
        SecPoints.append( [] )

    for i in range(triangles.shape[0]):

        t = triangles[i,:]
        a,b,c = t[0] , t[1] , t[2] 
        
        tri = [ a,b,c ] 

        try:
            sector = common_sector([points[x,:] for x in tri ], centroid)

        except ValueError:
            continue
        
        #print(tri)
        SCs[sector-1].insert( tri )
        SCs[sector-1].assign_filtration(tri , filtration=np.max( dists[[a,b,c]] ))
    
        e1 = [a,b]
        e2 = [a,c]
        e3 = [b,c]
    
        SCs[sector-1].insert( e1 )
        SCs[sector-1].assign_filtration(e1 , filtration=np.max( dists[[a,b]] ))
    
        SCs[sector-1].insert( e2 )
        SCs[sector-1].assign_filtration(e2 , filtration=np.max( dists[[a,c]] ))
    
        SCs[sector-1].insert( e3 )
        SCs[sector-1].assign_filtration(e3 , filtration=np.max( dists[[b,c]] ))
        
    for p in range(points.shape[0]):

        sector = which_sector(points[p,:] , centroid)

        SecPoints[sector-1].append( p )
    
        SCs[sector-1].insert([p])
        SCs[sector-1].assign_filtration([p] , filtration = dists[p] )


    res = {}
    
    for i,st in enumerate(SCs):
        st.compute_persistence()
        
        dgm0 = st.persistence_intervals_in_dimension(0)
        dgm1 = st.persistence_intervals_in_dimension(1)

        gens = st.lower_star_persistence_generators()

        if st.num_vertices() > 0:

            # points in this sector
            thisPoints = points[SecPoints[i], :]
            quantiles = quantiles_of_distance(thisPoints, centroid, which_quantiles)
    
            ordering = np.argsort(dists[SecPoints[i]]) # sort them by distance
    
            SecP = np.array(SecPoints[i], dtype = int)
    
            radial_charge = np.cumsum( potentials[SecP[ordering]] ) # potentials ordered by closest to farthest from centroid, cumulative sum
    
            significant_entries = [ np.floor( x * thisPoints.shape[0] ) for x in which_quantiles] # entries of radial charge corresponding to the desired quantiles
            significant_entries = np.array(significant_entries, dtype = int)
    
            cumulative_charge_at_quantiles = radial_charge[ significant_entries ] 
        else: 
            quantiles = [ 0.0 ] * len(which_quantiles)
            cumulative_charge_at_quantiles = np.zeros( ( len(which_quantiles) ) )

        res[i+1] = [ quantiles, cumulative_charge_at_quantiles, dgm0 , dgm1, gens ]


with open(os.path.join( './data/sectors/sublevelset_filtrations/train_set/', filename ) , 'wb') as out_file:
    pickle.dump(res , out_file)




    