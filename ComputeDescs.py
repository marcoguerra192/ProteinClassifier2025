import sys,os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK
from src.descriptors import AlphaDiag, PersImagesVectorize

from ripser import ripser
from persim import plot_diagrams

from gudhi import RipsComplex, AlphaComplex, plot_persistence_diagram
from gudhi.representations import ProminentPoints
from gudhi.representations import PersistenceImage

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

import joblib

import csv

# number of prominent points to keep for each dimension for each PD
Num_Prominent = 200
# resolution of each axis for the persistent images
PersImPoints = 5

source = DataSource( './data/train_set/', base_path='./data/train_set')
N_Files = len(list(source))

# vector of labels
labels = []


data = np.zeros( (3*(PersImPoints**2), N_Files) )

# read the labels
Truth = {}
with open('./data/train_set.csv', 'r') as csvfile:
    truths = csv.reader(csvfile)

    next(truths, None) # skip first row, it's a header
    for t in truths:
        Truth[t[0]] = int(t[1])

source = DataSource( './data/train_set/', base_path='./data/train_set')


for j,s in enumerate(source):

    print(j+1, 'out of', N_Files, flush=True)
    print(s, flush=True)

    # Find the label for the protein we are reading
    filename = os.path.basename(s)
    filename = os.path.splitext(filename)[0]
    labels.append(Truth[filename])

    points = read_vertices_VTK(s , out_var=True).tolist()

    Dgm0, Dgm1, Dgm2 = AlphaDiag(points, N_Prominent = Num_Prominent)

    Img = PersImagesVectorize(Dgm0, Dgm1, Dgm2, res = PersImPoints)

    data[:,j] = Img

# save descriptors
np.save('PersImages.npy', data.T)
np.save('Labels.npy', np.array(labels) )

    