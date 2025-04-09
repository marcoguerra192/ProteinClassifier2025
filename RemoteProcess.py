import sys,os
from pathlib import Path
import numpy as np

from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK, convertVTK_to_numpy
from src.descriptors.distance_dist import distances_from_point
from src.descriptors.distance_dist import centroid as Centroid

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from gudhi import SimplexTree
from gudhi import RipsComplex, AlphaComplex, plot_persistence_diagram

import joblib

import csv


source = DataSource( './data/data/test_set/', base_path='./data/data/test_set')
N_Files = len(list(source))

source = DataSource( './data/data/test_set/', base_path='./data/data/test_set/')


for j,s in enumerate(source):

    print(j+1, 'out of', N_Files, flush=True)
    print(s, flush=True)

    points, triangles, potentials, norm_potentials = convertVTK_to_numpy(s)

    filename = os.path.basename(s)

    output_file = os.path.join( './data/data/test_set_Numpy/', filename )

    np.savez(output_file, points=points, triangles=triangles, potentials=potentials, norm_potentials=norm_potentials)

    centroid = Centroid(points)
    dists = distances_from_point(points, centroid)
    st = SimplexTree()

    for i in range(triangles.shape[0]):

        t = triangles[i,:]
        a,b,c = t[0] , t[1] , t[2] 
        
        tri = [ a,b,c ] 
        #print(tri)
        st.insert( tri )
        st.assign_filtration(tri , filtration=np.max( dists[[a,b,c]] ))
    
        e1 = [a,b]
        e2 = [a,c]
        e3 = [b,c]
    
        st.insert( e1 )
        st.assign_filtration(e1 , filtration=np.max( dists[[a,b]] ))
    
        st.insert( e2 )
        st.assign_filtration(e2 , filtration=np.max( dists[[a,c]] ))
    
        st.insert( e3 )
        st.assign_filtration(e3 , filtration=np.max( dists[[b,c]] ))
        
    for p in range(points.shape[0]):
    
        st.insert([p])
        st.assign_filtration([p] , filtration = dists[p] )

    st.compute_persistence()
    dgm0 = st.persistence_intervals_in_dimension(0)
    dgm1 = st.persistence_intervals_in_dimension(1)
    dgm2 = st.persistence_intervals_in_dimension(2)

    gens = st.lower_star_persistence_generators()

    output_file2 = os.path.join( './data/data/sublevelset_filtrations/test_set/', filename )

    np.savez(output_file2, dgm0=dgm0, dgm1=dgm1, dgm2=dgm2, gens=gens, allow_pickle=True)




    