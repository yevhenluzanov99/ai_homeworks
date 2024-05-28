import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_array = np.concatenate([iris["data"], iris["target"].reshape(-1, 1)], axis=1)
features, classes = iris["feature_names"], iris["target_names"].tolist()
df = pd.DataFrame.from_records(iris_array, columns=features + ["class"])
x = df.iloc[:, [0, 1, 2, 3]].values
#metrics for Birch

sil_scores = []
ch_scores = []
db_scores = []

#2.1. Create Birch model
#2.2. Perform Hyperparameter Tuning of the created models with custom code
for i in range(1, 11):
    birch = Birch(n_clusters=i, branching_factor=50 )
    birch.fit(x)
    if i > 1:  # we dont use 1 cluster 
        y_birch = birch.predict(x)
        sil_scores.append(silhouette_score(x, y_birch))
        ch_scores.append(calinski_harabasz_score(x, y_birch))
        db_scores.append(davies_bouldin_score(x, y_birch))
    else:
        sil_scores.append(np.nan)  #  NaN value for 1 cluster
        ch_scores.append(np.nan)
        db_scores.append(np.nan)



#2.3. Plot results of Hyperparameter Tuning of each model
scores = { 'Silhouette Score': sil_scores, 'Calinski-Harabasz Score': ch_scores, 'Davies-Bouldin Score': db_scores}
plot_titles = list(scores.keys())
plot_colors = [ 'purple', 'orange', 'green']
plot_ylabel = [ 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
plt.figure(figsize=(16, 4))
for i, (title, data, color, ylabel) in enumerate(zip(plot_titles, scores.values(), plot_colors, plot_ylabel), 1):
    plt.subplot(1, 4, i)
    plt.plot(range(1, 11), data, marker='o', color=color)
    plt.title(title)
    plt.xlabel('Clusters Number')
    plt.ylabel(ylabel)
plt.tight_layout()
plt.show()


#BEst cluster number is 3
birch = Birch(n_clusters=3,branching_factor=50)
y_birch = birch.fit_predict(x)
#2.4. Plot the best clusters produced by each model
plt.scatter(x[y_birch == 0, 0], x[y_birch == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(x[y_birch == 1, 0], x[y_birch == 1, 1], s=100, c='orange', label='Iris-versicolour')
plt.scatter(x[y_birch == 2, 0], x[y_birch == 2, 1], s=100, c='green', label='Iris-virginica')

plt.legend()
plt.show()

#3d plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[y_birch == 0, 0], x[y_birch == 0, 1], x[y_birch == 0, 2], s=100, c='purple', label='Iris-setosa')
ax.scatter(x[y_birch == 1, 0], x[y_birch == 1, 1], x[y_birch == 1, 2], s=100, c='orange', label='Iris-versicolour')
ax.scatter(x[y_birch == 2, 0], x[y_birch == 2, 1], x[y_birch == 2, 2], s=100, c='green', label='Iris-virginica')
ax.set_title('3D Visualization of Iris Clusters')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel(features[2])
ax.legend()
plt.show()