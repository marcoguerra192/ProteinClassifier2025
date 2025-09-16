import sys,os
from pathlib import Path
import numpy as np

from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK, convertVTK_to_numpy
from src.descriptors.distance_dist import quantiles_of_distance, distances_from_point
from src.descriptors.distance_dist import centroid as Centroid
from src.descriptors.dscs_driver import process_spherical_sectors_descriptors, compute_spherical_sectors_descs

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from gudhi import SimplexTree
from gudhi import RipsComplex, AlphaComplex, plot_persistence_diagram

import joblib

import csv
import pickle


compute_spherical_sectors_descs('./data/data/test_set/' , './data/sectors/sublevelset_filtrations/test_set/')
