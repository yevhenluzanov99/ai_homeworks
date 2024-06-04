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

#1.1. Create Linear Regression model
regr = linear_model.LinearRegression()
#1.2. Perform Hyperparameter Tuning of the created model manually (if possible)
#No tuning for this method


# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred_test = regr.predict(x_test)
#1.3. Compute performance metrics for train & test datasets
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred_test))
#1.4. Extract coefficient and intercept of optimized Linear Function
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)



#1.5. Use extracted coefficient and intercept to plot optimized Linear Function together with original data points from dataset
#the result function
def f(x):
    return regr.coef_ * x + regr.intercept_
graph(f, x, x_test, y_test)