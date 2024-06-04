import matplotlib.pyplot as plt
import numpy as np


def graph(formula, x_range, x_test, y_test):  
    x = np.array(x_range)  
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)  
    plt.scatter(x_test, y_test, color="black")
    plt.show()  