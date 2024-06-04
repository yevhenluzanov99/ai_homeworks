import pandas as pd
from functions import graph
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv('homework7/icecream.csv')
df.columns = [col.lower() for col in df.columns]
x = df.drop(columns=['revenue'])
y = df['revenue']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

param_grid = {
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'n_iter_no_change': [5, 10, 20, 30,40 ]
}

#1.1. Create Linear Regression model
regr = linear_model.SGDRegressor(early_stopping=True, penalty=None, random_state=9)
#1.2. Perform Hyperparameter Tuning of the created model manually (if possible)
grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=5)
# Train the model using the training sets
grid_search.fit(x_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
best_sgd = grid_search.best_estimator_
best_sgd.fit(x_train, y_train)


#1.3. Compute performance metrics for train & test datasets
y_test_pred = best_sgd.predict(x_test)
print("Test Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test Coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))
y_train_pred = best_sgd.predict(x_train)
print("Train Mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train Coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

#1.4. Extract coefficient and intercept of optimized Linear Function
print("Coefficients: \n", best_sgd.coef_)
print("Intercept: \n", best_sgd.intercept_)
def f(x):
    return best_sgd.coef_ * x + best_sgd.intercept_
graph(f, x, x_test, y_test)




'''
check possible values for hyperparameters and their r2 score

results = grid_search.cv_results_
for i in range(len(results['params'])):
    model = grid_search.estimator.set_params(**results['params'][i])
    model.fit(x_train, y_train)
    intercept = model.intercept_
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"Parameters: {results['params'][i]}")
    print(f"Intercept: {intercept}")
    print(f"Train R^2 score: {r2_train}")
    print(f"Test R^2 score: {r2_test}\n")

'''