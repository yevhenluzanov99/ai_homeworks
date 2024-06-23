
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
import torch
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns




def set_seed(seed):
    """
    The function that sets a random seed to ensure the reproducibility of any randomized processes
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

class FlattenTransform():

    def __call__(self, tensor):
        return tensor.view(-1)
    

def create_dataloaders():
    """
    Create dataloaders for training and testing datasets.

    Returns:
        train_loader (torch.utils.data.DataLoader): Dataloader for the training dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader for the testing dataset.
        train_metrics (list): List to store training metrics.
        test_metrics (list): List to store testing metrics.
    """
    mnist_transform = transforms.Compose([transforms.ToTensor(),                # convert the image to a tensor
                                      FlattenTransform()])                  # flatten the tensor
    train_dataset = datasets.MNIST(root = './data', train = True, transform = mnist_transform, download = True)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = mnist_transform, download = True)

    train_loader = DataLoader(dataset = train_dataset, batch_size = 1000, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1000, shuffle = True)
    train_metrics = []
    test_metrics = []
    return train_loader, test_loader, train_metrics, test_metrics


import logging
import logging.handlers
import os

# Define the custom logging formatter
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def create_custom_logger(name,  log_file="application.log", log_level=logging.DEBUG):
    """
    Creates a custom logger with the specified name, log directory, log file, and log level.

    Parameters:
        name (str): Name of the logger.
        log_directory (str): Directory where log files will be stored.
        log_file (str): Name of the log file.
        log_level (int): Logging level.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Ensure the log directory exists
    log_directory = "homework9/logs"
    
    # Define the log file path
    log_file_path = os.path.join(log_directory, log_file)
    
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path)
    
    # Set level for handlers
    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)
    
    # Create formatters and add them to handlers
    c_format = CustomFormatter()
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times in interactive environments
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def plot_confusion_matrix(labels, preds, title):
    """
    Plots a confusion matrix using the labels and predictions.

    Parameters:
    labels (array-like): The true labels.
    preds (array-like): The predicted labels.
    title (str): The title of the plot.

    Returns:
    None
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()

def plot_loss(train_metrics, test_metrics, epochs):
    """
    Plots the training and validation losses over the specified number of epochs.

    Args:
        train_metrics (list): List of training loss values.
        test_metrics (list): List of validation loss values.
        epochs (int): Number of epochs.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_metrics, label="Training Loss")
    plt.plot(range(epochs), test_metrics, label="Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.show()
