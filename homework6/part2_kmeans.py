import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_array = np.concatenate([iris["data"], iris["target"].reshape(-1, 1)], axis=1)
features, classes = iris["feature_names"], iris["target_names"].tolist()
df = pd.DataFrame.from_records(iris_array, columns=features + ["class"])
x = df.iloc[:, [0, 1, 2, 3]].values
#metrics for KMeans
wcss = []
sil_scores = []
ch_scores = []
db_scores = []

#2.1. Create kMeans model
#2.2. Perform Hyperparameter Tuning of the created models with custom code
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
    if i > 1:  # we dont use 1 cluster 
        y_kmeans = kmeans.predict(x)
        sil_scores.append(silhouette_score(x, y_kmeans))
        ch_scores.append(calinski_harabasz_score(x, y_kmeans))
        db_scores.append(davies_bouldin_score(x, y_kmeans))
    else:
        sil_scores.append(np.nan)  #  NaN value for 1 cluster
        ch_scores.append(np.nan)
        db_scores.append(np.nan)



#2.3. Plot results of Hyperparameter Tuning of each model
scores = {'Elbow Method': wcss, 'Silhouette Score': sil_scores, 'Calinski-Harabasz Score': ch_scores, 'Davies-Bouldin Score': db_scores}
plot_titles = list(scores.keys())
plot_colors = ['blue', 'purple', 'orange', 'green']
plot_ylabel = ['WCSS', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
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
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
#2.4. Plot the best clusters produced by each model
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.legend()
plt.show()
#3d plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], x[y_kmeans == 0, 2], s=100, c='purple', label='Iris-setosa')
ax.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], x[y_kmeans == 1, 2], s=100, c='orange', label='Iris-versicolour')
ax.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], x[y_kmeans == 2, 2], s=100, c='green', label='Iris-virginica')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='red', label='Centroids', marker='*')

ax.set_title('3D Visualization of Iris Clusters')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel(features[2])
ax.legend()
plt.show()