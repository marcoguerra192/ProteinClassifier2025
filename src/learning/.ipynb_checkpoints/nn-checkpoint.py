''' 
/* SHREC 2025
Marco Guerra

*/
'''

# NN models

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 50) 
        self.dr = nn.Dropout(p=.25) # dropout layer
        self.fc2 = nn.Linear(50, 50) 
        self.fc3 = nn.Linear(75,num_classes) 
        self.fc4 = nn.Linear(50, num_classes)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        x = self.dr(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dr(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.dr(x)
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)  # No softmax, since CrossEntropyLoss applies it internally
        return x

def train(N_epochs,model,criterion, optimizer, train_loader, val_loader, save_path, patience=25 ):

    best_loss = float('inf')
    counter = 0

    train_losses = []
    #true_tr_losses = []
    val_losses = []

    for epoch in range(N_epochs):  # Train for up to N_epochs epochs

        running_train_loss = 0.0
        
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Loss
        
        model.eval()

        # ### TRUE LOSS
        # true_tr_loss = 0.0
        # with torch.no_grad():
        #     for X_batch, y_batch in train_loader:
        #         outputs = model(X_batch)
        #         loss = criterion(outputs, y_batch)
        #         true_tr_loss += loss.item()

        # true_tr_loss /= len(train_loader)
        # true_tr_losses.append(true_tr_loss)

        
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
    
        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
            torch.save(model.state_dict(), save_path)  # Save best model
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break  # Stop training

        # Live plot update
        clear_output(wait=True)

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        #plt.plot(true_tr_losses, label="True Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()


    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")


def predict(model, val_loader, device):

    model.eval()  # Set model to evaluation mode
    y_pred = []
    
    with torch.no_grad():  # No gradient calculation needed
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)  # Get class with highest probability
            y_pred.extend(predicted.cpu().numpy())

    return y_pred




