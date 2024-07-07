import numpy as np

from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import functions as f

mylogger = f.create_custom_logger("mylogger", "connected_autoencoder.log")
metrics_path = "homework11/metrics/connected_autoencoder.pkl"
model_path = "homework11/models/connected_autoencoder.pth"


# Model
class FCAutoEncoder(nn.Module):
    """
    Fully Connected Autoencoder
    """

    def __init__(self, in_dim, out_dim):
        super(FCAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 324),
            nn.BatchNorm1d(324),
            nn.LeakyReLU(),
            nn.Linear(324, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 324),
            nn.BatchNorm1d(324),
            nn.LeakyReLU(),
            nn.Linear(324, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x1 = self.encoder(x)
        output = self.decoder(x1)
        return output


f.set_seed(9)

# Use GPU's CUDA cores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
# Initialization of Model, Cost function and Optimizer
# Input & Output Dimensions of the Model
# input dimension = n of features in dataset
# output dimension = n of features in dataset
in_dim, out_dim = 28 * 28, 28 * 28
# Learning Rate
lr = 0.001
# n of iterations of Optimizer
epochs = 40
mylogger.info(f"learning rate: {lr}, epochs: {epochs}")
# Model
model = FCAutoEncoder(in_dim, out_dim)
model = model.to(device)
# Cost Function
criterion = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loader, test_loader, train_metrics, test_metrics = f.create_mnist_dataloaders_for_conected()


def train_autoencoder():
    for epoch in range(epochs):

        ### Training ###
        train_loss = []
        model.train()

        for x_tr, _ in train_loader:
            # Pass data to GPU's CUDA cores
            x_tr = x_tr.to(device)

            # Compute predictions
            outputs = model(x_tr)
            # Compute Loss for predictions
            loss = criterion(outputs, x_tr)
            # Remove previous gradients
            optimizer.zero_grad()
            # Compute current gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            # Save training Loss
            train_loss.append(loss.item())

        # Get mean Training Loss per epoch
        train_mean_loss = sum(train_loss) / len(train_loss)
        train_metrics.append(train_mean_loss)

        ### Validation ###
        test_loss = []
        model.eval()

        with torch.no_grad():
            for x_ts, _ in test_loader:

                x_ts = x_ts.to(device)

                # Compute predictions
                outputs = model(x_ts)
                # Compute Loss for predictions
                loss = criterion(outputs, x_ts)
                # Save validation Loss
                test_loss.append(loss.item())

        # Get mean of  Validation Loss per epoch
        test_mean_loss = sum(test_loss) / len(test_loss)
        test_metrics.append(test_mean_loss)
        mylogger.info(
            f"Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}"
        )
        print(
            f"Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}"
        )
    torch.save(model.state_dict(), model_path)
    mylogger.info(f"Model saved to {model_path}")


"""
train_autoencoder()
with open(metrics_path, "wb") as file:
    pickle.dump((train_metrics, test_metrics),file)
"""
# Final predictions
with open(metrics_path, "rb") as file:
    train_metrics, test_metrics = pickle.load(file)

loaded_model = FCAutoEncoder(in_dim, out_dim)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model = loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():

    ### Test dataset ###
    x_ts, _ = next(iter(test_loader))
    x = x_ts[:10]
    x = x.to(device)
    x_pred = loaded_model(x)

x = x.view(10, 28, 28).cpu().numpy()
x_pred = x_pred.view(10, 28, 28).detach().cpu().numpy()
mse_values = []

with torch.no_grad():
    for x_ts, _ in test_loader:
        x_ts = x_ts.to(device)

        # Compute predictions
        outputs = loaded_model(x_ts)
        mse = nn.MSELoss(reduction="none")(outputs, x_ts)  # MSE per image
        mse_values.extend(mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy())

# Calculate overall mean MSE
mean_mse = np.mean(mse_values)

mylogger.info(f"Global Mean Squared Error (MSE) across all test images: {mean_mse:.6f}")
print(f"Global Mean Squared Error (MSE) across all test images: {mean_mse:.6f}")


mse_values = []

# Evaluate MSE for the first 10 images
with torch.no_grad():
    for i, (x_ts, _) in enumerate(test_loader):
        if i >= 1:  # Process only the first batch of test data
            break
        x_ts = x_ts.to(device)

        # Compute predictions
        outputs = loaded_model(x_ts)

        # Compute MSE for each image in the batch
        mse = nn.MSELoss(reduction="none")(outputs, x_ts)

        # Calculate mean MSE per image (assuming 28x28 images)
        mse_per_image = mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy()

        # Append to list of MSE values
        mse_values.extend(mse_per_image)

        # Convert images and predictions to numpy arrays
        x = x_ts.view(x_ts.size(0), 28, 28).cpu().numpy()
        x_pred = outputs.view(outputs.size(0), 28, 28).cpu().numpy()

        # Print MSE for each image
        for j in range(min(x_ts.size(0), 10)):  # Print MSE for up to 10 images
            mylogger.info(f"Image {j+1}: MSE = {mse_per_image[j]:.6f}")
            print(f"Image {j+1}: MSE = {mse_per_image[j]:.6f}")
# 1.3. Plot results of model training
f.plot_loss(train_metrics, test_metrics, epochs)

# 1.4. Plot original and reconstructed images
f.plot_pictures(x, x_pred)