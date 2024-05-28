import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
iris_array = np.concatenate([iris["data"], iris["target"].reshape(-1, 1)], axis=1)
features, classes = iris["feature_names"], iris["target_names"].tolist()
df = pd.DataFrame.from_records(iris_array, columns=features + ["class"])
x = df.iloc[:, [0, 1, 2, 3]].values

#metrics for DBSCAN
sil_scores = []
ch_scores = []
db_scores = []

#2.1. Create kMeans model
#2.2. Perform Hyperparameter Tuning of the created models with custom code
for eps in np.linspace(0.1, 2.0, num=20):
    dbscan = DBSCAN(eps=eps, algorithm='brute') 
    y_dbscan = dbscan.fit_predict(x)
    unique_labels = set(y_dbscan)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Исключаем шумовой кластер (-1)
    
    if len(np.unique(y_dbscan)) > 1:
        sil_scores.append(silhouette_score(x, y_dbscan))
        ch_scores.append(calinski_harabasz_score(x, y_dbscan))
        db_scores.append(davies_bouldin_score(x, y_dbscan))
    else:
        sil_scores.append(np.nan)
        ch_scores.append(np.nan)
        db_scores.append(np.nan)


scores = {'Silhouette Score': sil_scores, 'Calinski-Harabasz Score': ch_scores, 'Davies-Bouldin Score': db_scores}
plot_titles = list(scores.keys())
plot_colors = ['purple', 'orange', 'green']
plot_ylabel = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
#2.3. Plot results of Hyperparameter Tuning of each model
plt.figure(figsize=(12, 4))
for i, (title, data, color, ylabel) in enumerate(zip(plot_titles, scores.values(), plot_colors, plot_ylabel), 1):
    plt.subplot(1, 3, i)
    plt.plot(np.linspace(0.1, 2.0, num=20), data, marker='o', color=color)
    plt.title(title)
    plt.xlabel('Epsilon')
    plt.ylabel(ylabel)
plt.tight_layout()
plt.show()



#2.4. Plot the best clusters produced by each model

best_eps_index = np.nanargmax(sil_scores)  # Находим индекс лучшего значения Silhouette Score
best_eps = np.linspace(0.1, 2.0, num=20)[best_eps_index]  # Находим соответствующий epsilon
best_dbscan = DBSCAN(eps=best_eps, min_samples=5)
y_best_dbscan = best_dbscan.fit_predict(x)
plt.scatter(x[y_best_dbscan == 0, 0], x[y_best_dbscan == 0, 1], s=100, c='purple', label='Cluster 0')
plt.scatter(x[y_best_dbscan == 1, 0], x[y_best_dbscan == 1, 1], s=100, c='orange', label='Cluster 1')
plt.legend()
plt.show()


#3d plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
unique_labels = np.unique(y_best_dbscan)
colors = ['purple', 'orange', 'green', 'blue', 'red', 'yellow', 'cyan', 'magenta']

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black'
        label_name = 'Noise'
    else:
        label_name = f'Cluster {label}'

    ax.scatter(x[y_best_dbscan == label, 0], x[y_best_dbscan == label, 1], x[y_best_dbscan == label, 2], s=100, c=color, label=label_name)

ax.set_title('3D Visualization of DBSCAN Clusters')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.legend()
plt.show()