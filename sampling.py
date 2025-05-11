from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from torch.utils.data import Dataset
from diffusion_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader




# hyperparameters

# diffusion hyperparameters
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 256 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 128
save_dir = 'weights/'

# training hyperparameters
batch_size = 32
n_epoch = 100
lrate=1e-3


# construct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)



# Load real datasets
X_train = np.load('wind_train_3D16X16.npy')
y_train = np.load('wind_label_train_3D16X16.npy')
X_val = np.load('wind_val_3D16X16.npy')
y_val = np.load('wind_label_val_3D16X16.npy')

# Convert numpy arrays to PyTorch tensors and PERMUTE properly
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 3, 16, 16)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

perturbed_train_dataset = PrecomputedPerturbedDataset(train_dataset, timesteps)
perturbed_val_dataset = PrecomputedPerturbedDataset(val_dataset, timesteps)
train_loader = DataLoader(perturbed_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(perturbed_val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)



criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Training loop
num_epochs = 250
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    nn_model.train()
    running_loss = 0.0
    for x_pert, t, noise in tqdm(train_loader): 
        x_pert = x_pert.to(device).squeeze(1)
        t = t.to(device)
        noise = noise.to(device)
        optimizer.zero_grad()
        pred_noise = nn_model(x_pert, t / timesteps)
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_pert, t, noise in tqdm(val_loader):
            x_pert = x_pert.to(device).squeeze(1)
            t = t.to(device)
            noise = noise.to(device)
            pred_noise = nn_model(x_pert, t / timesteps)
            loss = F.mse_loss(pred_noise, noise)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    # save model periodically
    if epoch%4==0 or epoch == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"model_{epoch}.pth")
        print('saved model at ' + save_dir + f"model_{epoch}.pth")


plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save to present directory
plt.savefig("training_validation_loss.png")