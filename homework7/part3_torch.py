import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from functions import graph
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 2.3. Plot results of model training
def plot_train_result(epochs, train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()


def compute_cost(theta_1, x, y_true):
    n = len(y_true)
    y_pred = theta_1 * x
    cost = 1 / n * np.sum(np.square(y_true - y_pred))

    return cost


# Moving average
def moving_average(data, window_size):
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode="linear_ramp")
    windows = sliding_window_view(padded_data, window_size)
    moving_avg = np.mean(windows, axis=1)

    if window_size % 2 == 0:
        return moving_avg[:-1]
    else:
        return moving_avg


def set_seed(seed):
    """
    The function that sets a random seed to ensure the reproducibility of any randomized processes
    """

    np.random.seed(seed)
    torch.manual_seed(seed)


# Model
class LinearRegression_torch(nn.Module):
    """
    Linear Regression model (single neuron Neural Network)
    """

    def __init__(self, in_dim, out_dim):
        super(LinearRegression_torch, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        output = self.linear(x)
        return output


df = pd.read_csv("homework7/icecream.csv")
df.columns = [col.lower() for col in df.columns]
x = df.drop(columns=["revenue"])
y = df["revenue"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

x_train = torch.from_numpy(x_train.values.astype(np.float32))
y_train = torch.from_numpy(y_train.values.astype(np.float32))
x_test = torch.from_numpy(x_test.values.astype(np.float32))
y_test = torch.from_numpy(y_test.values.astype(np.float32))

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train, shuffle=True)
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test, batch_size=200, shuffle=False)
set_seed(9)
in_dim, out_dim = 1, 1  # input and output dimensions of the Model
# n of iterations of Optimizer
epochs = 100
# Model
model = LinearRegression_torch(in_dim, out_dim)
# Cost Function
criterion = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
train_metrics = []
test_metrics = []
for epoch in range(epochs):

    ### Training ###
    train_loss = []
    model.train()

    for i, (x_tr, y_tr) in enumerate(train_loader):

        # Compute predictions
        outputs = model(x_tr.reshape(-1, 1))
        # Compute Loss for predictions
        loss = criterion(outputs, y_tr.reshape(-1, 1))
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

            # Compute predictions
            outputs = model(x_ts.reshape(-1, 1))
            # Compute Loss for predictions
            loss = criterion(outputs, y_ts.reshape(-1, 1))
            # Save validation Loss
            test_loss.append(loss.item())

    # Get mean of  Validation Loss per epoch
    test_mean_loss = sum(test_loss) / len(test_loss)
    test_metrics.append(test_mean_loss)

    print(
        f"Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}"
    )


# 2.3. Plot results of model training
plot_train_result(epochs, train_metrics, test_metrics)

# 2.4. Compute final performance metrics for train & test datasets
y_train_pred = model(x_train.reshape(-1, 1)).detach().numpy()
y_test_pred = model(x_test.reshape(-1, 1)).detach().numpy()
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Final Performance Metrics:")
print(f"Train R^2: {r2_train:.4f}, Train MSE: {mse_train:.4f}")
print(f"Test R^2: {r2_test:.4f}, Test MSE: {mse_test:.4f}")

# 2.5. Extract coefficient and intercept of optimized Linear Function
coef = model.linear.weight.item()
intercept = model.linear.bias.item()
print(f"Coefficient: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")


# 2.6. Use extracted coefficient and intercept to plot optimized Linear Function together with original data points from dataset
def f(x):
    return coef * x + intercept


graph(f, x, x_test, y_test)


# 2.7. Plot Cost-Residual Planne
def plot_cost_resudual_plane():
    x_len = x_train.shape[0]
    # Generate a range of values for theta_1
    theta_vals = np.linspace(925, 1175, x_len)
    # Compute the cost for each value of theta
    J_vals = np.array(
        [compute_cost(i, x_train.numpy(), y_train.numpy()) for i in theta_vals]
    )
    # Compute the residuals for each prediction
    y_pred = model(x_train.reshape(-1, 1)).detach().numpy()
    res_vals = np.square(y_train.numpy().reshape(-1, 1) - y_pred)
    # Reshape residuals  into a vector
    res_vals = res_vals.reshape(len(res_vals))
    # Apply moving average to smooth the noise in residuals
    smooth_res_vals = moving_average(res_vals, window_size=5)
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # Plane of Cost Function
    ax.plot(
        theta_vals,
        J_vals,
        label=" plane of Cost Function (MSE)",
        color="blue",
        linewidth=1,
    )
    # Plane of Cost Function and Residuals
    ax.plot(
        theta_vals,
        smooth_res_vals + J_vals,
        label="Cost-Residual plane (Gradient Descent works here)",
        color="red",
        linewidth=1,
    )
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$J(\theta)$")
    plt.title("Plane of Cost Function")
    plt.grid(True, zorder=-1)
    plt.legend()
    plt.show()


plot_cost_resudual_plane()
