# File part1.py
    SVM Model 

# File part2_kmeans.py

    kmeans cluster model

# File part3_dbscan.py

    dbscan cluster model

# File part4_birch.py
    birch cluster model

Duaring all experiments i have found that:
    1. The SVM model works really well with pictures and vectors. Its prediction accuracy is very high, achieving around 98-99% true prediction on the training dataset and 97-98% on the test dataset
    2. Duaring cluster`s test i have found thats in case of  kmeans cluster and birch cluster the result is almost corect. Dbscan dont answer for the question of best cluster count. This is due to the fact that the method analyzes the density of data and in our case we analyze different subtypes of the same type of flower, which are visually very similar and differ only in small details. I think it is better to use this method when the data is well separated from each other, or to consider more global objects.
    3. The best cluster, i think, is kmeans. Because we dont have a huge dataset , birch is better on huge datasets, dbscan is better on more massive objects

