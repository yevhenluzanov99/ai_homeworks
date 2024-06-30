import torch
from torch import nn
import functions as f
import pickle
import numpy as np
from torchvision import datasets, transforms



mylogger = f.create_custom_logger("mylogger", "CNN_mnist.log")
metrics_path = "homework10/metrics/CNN_mnist.pkl"
model_path = "homework10/models/CNN_mnist.pth"

class CNN(nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 16, kernel_size = 5, stride = 1, padding = 2),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(kernel_size = 2))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(kernel_size = 2))

        self.fc_output = nn.Sequential(nn.Linear(32 * 8 * 8, out_dim))

        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'leaky_relu')

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = self.fc_output(x2.view(x2.size(0), -1))

        return output

train_loader, test_loader, train_metrics, test_metrics = f.create_mnist_dataloaders()
f.set_seed(9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')
in_dim, out_dim = 1, 10
lr = 0.001
epochs = 10
# Model
mylogger.info(f"learning rate: {lr}, epochs: {epochs}")
model = CNN(in_dim, out_dim)
model = model.to(device)

# Cost Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
def train_cnn():
    for epoch in range(epochs):

        ### Training ###
        train_loss = []
        model.train()

        for x_tr, y_tr in train_loader:
            # Pass data to GPU's CUDA cores
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)

            # Compute predictions
            outputs = model(x_tr)
            # Compute Loss for predictions
            loss = criterion(outputs, y_tr)
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
            for x_ts, y_ts in test_loader:

                x_ts, y_ts = x_ts.to(device), y_ts.to(device)

                # Compute predictions
                outputs = model(x_ts)
                # Compute Loss for predictions
                loss = criterion(outputs, y_ts)
                # Save validation Loss
                test_loss.append(loss.item())

        # Get mean of  Validation Loss per epoch
        test_mean_loss = sum(test_loss) / len(test_loss)
        test_metrics.append(test_mean_loss)
        mylogger.info(
            f"Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}"
        )
        print(f'Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}')
    torch.save(model.state_dict(), model_path)
    mylogger.info(f"Model saved to {model_path}")
'''
train_cnn()
with open(metrics_path, "wb") as file:
    pickle.dump((train_metrics, test_metrics),file)
'''
# Load train_metrics and test_metrics from file
with open(metrics_path, "rb") as file:
    train_metrics, test_metrics = pickle.load(file)


loaded_model = CNN(in_dim, out_dim)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model = loaded_model.to(device)
loaded_model.eval()
with torch.no_grad():

    ### Training dataset ###
    correct, total = 0, 0
    all_preds_tr = []
    all_labels_tr = []
    for x_tr, y_tr in train_loader:

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        y_pred_tr = torch.softmax(loaded_model(x_tr), dim = 1)
        y_pred_tr_class = torch.argmax(y_pred_tr, dim = 1)
        all_preds_tr.extend(y_pred_tr_class.cpu().numpy())
        all_labels_tr.extend(y_tr.cpu().numpy())
        # retrieve class with the highest probability
        correct += (y_pred_tr_class == y_tr).sum().item()
        total += y_tr.size(0)

    train_acc = correct / total
    mylogger.info(f"train_accuracy = {train_acc:.4f}")
    print(f'train_accuracy = {train_acc:.4f}')

    ### Test dataset ###
    correct, total = 0, 0
    all_preds_ts = []
    all_labels_ts = []
    for x_ts, y_ts in test_loader:

        x_ts, y_ts = x_ts.to(device), y_ts.to(device)
        y_pred_ts = torch.softmax(loaded_model(x_ts), dim = 1)
        # retrieve class with the highest probability
        y_pred_ts_class = torch.argmax(y_pred_ts, dim = 1)
        all_preds_ts.extend(y_pred_ts_class.cpu().numpy())
        all_labels_ts.extend(y_ts.cpu().numpy())
        correct += (y_pred_ts_class == y_ts).sum().item()
        total += y_ts.size(0)

    test_acc = correct / total
    mylogger.info(f"test_accuracy = {test_acc:.4f}")
    print(f'test_accuracy = {test_acc:.4f}')

# 1.3. Plot results of model training
f.plot_loss(train_metrics, test_metrics, epochs)
# 1.5. Plot Confusion Matrix for train & test datasets
# Plot Confusion Matrix for training dataset
f.plot_confusion_matrix(
    all_labels_tr, all_preds_tr, "Confusion Matrix for Training Dataset"
)
# Plot Confusion Matrix for test dataset
f.plot_confusion_matrix(
    all_labels_ts, all_preds_ts, "Confusion Matrix for Test Dataset"
)
