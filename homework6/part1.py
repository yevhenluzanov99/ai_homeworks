import numpy as np
import pandas as pd

from sklearn import datasets

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def print_metrics(y_true, y_pred, dataset_type="Dataset"):
    print(f"Performance metrics for {dataset_type}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred, target_names=classes))


# load dataset
iris = datasets.load_iris()

# concatenate feature valued and class values
iris_array = np.concatenate([iris["data"], iris["target"].reshape(-1, 1)], axis=1)

# extract feature name and class labels
features, classes = iris["feature_names"], iris["target_names"].tolist()

# tranform np.ndarray to pd.DataFrame for convenience
df = pd.DataFrame.from_records(iris_array, columns=features + ["class"])

x = df.drop(columns=["class"])

y = df["class"]
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

hparams_grid = {"C": [0.1, 0.5, 1], "kernel": ["linear", "poly", "sigmoid"]}
#1.1. Create SVM Classification model
svc = SVC(random_state=9)
#1.2. Perform Hyperparameter Tuning of the created model with Grid Search Cross-Validation
grid_search = GridSearchCV(estimator=svc, param_grid=hparams_grid, cv=5)

# Обучение модели с использованием GridSearchCV
grid_search.fit(x_train, y_train)
# 1.3 Plot results of Grid Search Cross-Validation
results = pd.DataFrame(grid_search.cv_results_)

# Create pivot table for the heatmap
scores = results.pivot_table(index="param_C", columns="param_kernel", values="mean_test_score")

plt.figure(figsize=(10, 7))
sns.heatmap(scores, annot=True, cmap="viridis")
plt.title("Grid Search CV Results")
plt.xlabel("Kernel")
plt.ylabel("C")
plt.show()


#1.4 Extract already trained the best model from Grid Search Cross-Validation
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

#1.5 Compute performance metrics for train & test datasets
print_metrics(y_train, y_train_pred, "Training Set")
print_metrics(y_test, y_test_pred, "Test Set")

#1.6 Plot Confusion Matrix for train & test datasets
labels_test = np.sort(np.unique(y_test))
cm_vals_test = confusion_matrix(y_test, y_test_pred, labels = labels_test)
cm_plot_test = ConfusionMatrixDisplay(confusion_matrix = cm_vals_test, display_labels = labels_test)
cm_plot_test.plot()
plt.show()


labels_train = np.sort(np.unique(y_train))
cm_vals_train = confusion_matrix(y_train, y_train_pred, labels = labels_train)
cm_plot_train = ConfusionMatrixDisplay(confusion_matrix = cm_vals_train, display_labels = labels_train)
cm_plot_train.plot()
plt.show()
