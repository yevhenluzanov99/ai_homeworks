import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from functions import preprocess_data

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.show()


df = preprocess_data()
x = df.drop(columns=["survived"])
y = df["survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

param_grid = {
    "eta0": [0.001, 0.01, 0.1, 1, 10, 100],
    "max_iter": [20, 50, 100, 200, 300],
    "n_iter_no_change": [5, 10, 20, 50, 100],
}

# 1.1.  Create Logistic Regression model
regr = linear_model.SGDClassifier(
    penalty=None,
    random_state=9,
    loss="log_loss",
    learning_rate="constant",
    early_stopping=True,
)
# 1.2. Perform Hyperparameter Tuning of the created model manually (if possible)
grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
best_sgd = grid_search.best_estimator_
best_sgd.fit(x_train, y_train)
# 1.3. Compute Accuracy for train & test datasets
y_pred_test = best_sgd.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test accuracy with best parameters: {accuracy_test}")
y_pred_train = best_sgd.predict(x_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Train accuracy : {accuracy_train}")
# 1.4. Plot Confusion Matrix for train & test datasets
plot_confusion_matrix(y_test, y_pred_test, classes=best_sgd.classes_)
