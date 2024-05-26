import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from homework5.custom_pickle import save_pickle

#function to plot the search grid results
def plot_search_results(grid):
    """
    Params:
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results["mean_test_score"]
    stds_test = results["std_test_score"]
    means_train = results["mean_train_score"]
    stds_train = results["std_train_score"]

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results["param_" + p_k].data == p_v))

    params = grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex="none", sharey="all", figsize=(20, 5))
    fig.suptitle("Score per parameter")
    fig.text(0.04, 0.5, "MEAN SCORE", va="center", rotation="vertical")
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1 :])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle="--", marker="o", label="test")
        ax[i].errorbar(x, y_2, e_2, linestyle="-", marker="^", label="train")
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()


df = pd.read_csv("homework5/heartRisk.csv")

df.columns = [col.lower() for col in df.columns]
x = df.drop(columns=["risk"])
y = df["risk"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
hparams_grid = {
    "n_estimators": [10, 50, 100],  # i check value 10,50,100  - best 100
    "max_depth": [5, 10, 20], # i check value 5,10,20, 30 - best 10 because its max
    "min_samples_split": [2, 5, 10],  #  i check value 2,5,10, 20 - best 2
    "min_samples_leaf": [2, 5],  #  i check value 2,9, 10, 20 - best 2
    "max_features": [9, "log2"], # i check value None, 9, sqrt, log2 - best  9 because its max or None
}
model = RandomForestRegressor(criterion="squared_error", random_state=9)
# Init Grid Search Cross-Validation
# Use cv = 5, scoring = 'neg_mean_squared_error' (MSE), verbose = 3 and return_train_score = True
gs_ds = GridSearchCV(
    model,
    hparams_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=3,
    return_train_score=True,
)

# Run Grid Search Cross-Validation, it is the most time consuming part
gs_ds.fit(x_train, y_train)

# Extract total results of Grid Search Cross-Validation
gs_results = gs_ds.cv_results_

# Visuzalize results of Grid Search Cross-Validation

params = gs_results["params"]
plot_search_results(gs_ds)
# Extract already trained the best model
my_best_model = gs_ds.best_estimator_

# Extract hyperparameters of already trained the best model
my_best_hparams = gs_ds.best_params_
print(my_best_hparams)
# Compute prediction scores
y_train_pred = my_best_model.predict(x_train)
y_test_pred = my_best_model.predict(x_test)
save_pickle(my_best_model, "homework5/trained_forest_tree.pickle")