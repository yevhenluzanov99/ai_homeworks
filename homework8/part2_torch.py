import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import  train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from functions import preprocess_data

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.show()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LogisticRegression_torch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegression_torch, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        output = self.sigmoid(z)
        return output


df = preprocess_data()
x = df.drop(columns=["survived"])
y = df["survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

x_train = torch.from_numpy(x_train.values.astype(np.float32))
y_train = torch.from_numpy(y_train.values.astype(np.float32))
x_test = torch.from_numpy(x_test.values.astype(np.float32))
y_test = torch.from_numpy(y_test.values.astype(np.float32))

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train, shuffle=True)
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test, batch_size=300, shuffle=True)
set_seed(9)
# 2.2. Perform Hyperparameter Tuning of the created model manually (play around)
in_dim, out_dim = 10, 1
lr = 1e-4
epochs = 100
# 2.1. Create Logistic Regression model
model = LogisticRegression_torch(in_dim, out_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_metrics = []
test_metrics = []

for epoch in range(epochs):
    train_loss = []
    model.train()
    for i, (x_tr, y_tr) in enumerate(train_loader):
        outputs = model(x_tr)
        loss = criterion(outputs, y_tr.reshape(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    train_mean_loss = sum(train_loss) / len(train_loss)
    train_metrics.append(train_mean_loss)

    test_loss = []
    model.eval()
    with torch.no_grad():
        for x_ts, y_ts in test_loader:
            outputs = model(x_ts)
            loss = criterion(outputs, y_ts.reshape(-1, 1))
            test_loss.append(loss.item())

    test_mean_loss = sum(test_loss) / len(test_loss)
    test_metrics.append(test_mean_loss)
    print(
        f"Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}"
    )

# 2.4. Compute Accuracy for train & test datasets
with torch.no_grad():
    y_pred_train = model(x_train)
    y_pred_train_class = y_pred_train.round()
    train_acc = (
        torch.sum(y_pred_train_class == y_train.reshape(-1, 1)) / y_train.shape[0]
    )
    print(f"train_accuracy = {train_acc:.4f}")

    y_pred_test = model(x_test)
    y_pred_test_class = y_pred_test.round()
    test_acc = torch.sum(y_pred_test_class == y_test.reshape(-1, 1)) / y_test.shape[0]
    print(f"test_accuracy = {test_acc:.4f}")


# create plot of your Training & Validation Losses (x axis -- n of epochs, y axis -- Training & Validation Losses).
# 2.3. Plot results of model training
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_metrics, label="Training Loss")
plt.plot(range(epochs), test_metrics, label="Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.show()
# Train Epoch: 98          [Train Loss]: 0.478839          [Test Loss]: 0.464019

# 2.5. Plot Confusion Matrix for train & test datasets
plot_confusion_matrix(y_test, y_pred_test_class, classes=["Not Survived", "Survived"])
