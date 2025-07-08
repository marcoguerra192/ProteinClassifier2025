import sys,os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK
from src.descriptors.dscs_driver import compute_descriptors

from src.utils import get_free_id, write_jsonl, clear_jsonl

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve

import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary as torch_summary

from IPython.display import display, clear_output

from src.learning.nn import SimpleNN, train, predict, Autoencoder

print('TRAINING RUN', flush=True)

dataEncAlpha_aug = np.load('./data/data/saved_descriptors/train_set/EncodedAlphaProminent_aug.npy')
dataS_aug = np.load('./data/sectors/saved_descriptors/train_set/AugmentedData.npy')
labels = np.load('./data/sectors/saved_descriptors/train_set/AugmentedLabels.npy')

dataEncAlpha_aug = np.load('./data/data/saved_descriptors/train_set/CleanAlphaProminent_aug.npy')
dataS_aug = np.load('./data/sectors/saved_descriptors/train_set/AugmentedData.npy')
labels = np.load('./data/sectors/saved_descriptors/train_set/AugmentedLabels.npy')

#print(dataS_aug.shape)
#print(dataEncAlpha_aug.shape)


data = np.concatenate( [dataS_aug , dataEncAlpha_aug] , axis = 1 )

print('Shape of data')
print(data.shape, flush=True)


StrShSp = StratifiedShuffleSplit(n_splits=15, train_size=0.8, random_state=42)

# get indices of split
train_idx, val_idx = next(StrShSp.split(data, labels))

train_data = data[train_idx,:]
train_labels = labels[train_idx]

val_data = data[val_idx]
val_labels = labels[val_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device = ',device)

X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_val = torch.tensor(val_data, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.long)

#WX_val = torch.tensor(data, dtype=torch.float32)
#Wy_val = torch.tensor(labels, dtype=torch.long)

print('Data Loader', flush=True)

# data loader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)
#whole_data = DataLoader(TensorDataset(WX_val, Wy_val), batch_size=128)

# Model, Loss, and Optimizer

m = train_data.shape[1]
c = np.unique(labels).shape[0]

print('Sizes=', m,c, flush=True)

# Model save path
trained_model_path = './trained_models/nn/remote_best_weights.pth'

model = SimpleNN(m, c)

summary = torch_summary(model,None)
print(summary, flush=True)

try: 
    pass
    #model.load_state_dict(torch.load(trained_model_path))
except:
    pass
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)  # Adam optimizer

epochs = 5000
# Early Stopping Parameters
patience = int(epochs/10)  # Stop if no improvement for 'patience' epochs

print('Start training', flush=True)
#quit()

train(epochs, model, criterion, optimizer, train_loader, val_loader, trained_model_path, patience)


print('Finished training')