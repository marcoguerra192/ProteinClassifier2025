# Naming Conventions

#### SHREC 2025
Marco Guerra

### Datasets

- `data` - the whole dataset
- `small_data` - a balanced subset of 900 proteins, where every label appears at least twice
- `micro_data` - a random subset of 25 proteins

### Trained models

#### nn
Neural networks. In each folder we aim to have a json file explaining hyperparameters

- `FF` - Simple vanilla feed-forward neural network
- `autoencoder` - An autoencoder instance to compress the descriptors
- `gnn` - An instance of Graph Neural Network. This uses pooling and message passing to classify data, enforcing rotational invariance

#### Random forest
Array of decision trees. In each folder, we would like to have a json file explaining hyperparameters

### Runs
Folder with json file(s) to record the experimental runs. For each we aim to record:

- Which model we used (naming as above)
- Which parameters/hyperparameters (possibily pointing to a json file)
- Which data we used it on (naming as above)
  The performance achieved
