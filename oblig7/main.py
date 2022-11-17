# %%
"""
## Oblig 7 - Unsupervised Learning

"""

# %%
"""
1. Using the UCI mushroom dataset, use k-means and a suitable cluster evaluation metric to determine the optimal number of clusters in the dataet. Note that this may not be necessarily two (edible V non-edible)

2. Plot the metric while increasing the number of clusters. e.g, k=2..30

"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from sklearn import decomposition
from sklearn.decomposition import PCA


# from sklearn.datasets.samples_generator import make_blobs


# %%
# import mushrooms dataset
mushrooms = pd.read_csv('agaricus-lepiota.csv')

mushrooms.head()


# %%

X = pd.get_dummies(mushrooms.drop('edibility', axis='columns'))

ssd = []
sc = []  # Sillhouette Coefficient, mean nearest-cluster distance

k = range(2, 20)  # (2,20) instead of (2,30)...time consuming to run every time


for n in k:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)
    sc.append(metrics.silhouette_score(
        X.values, kmeans.labels_, metric='euclidean'))

#y_kmeans = kmeans.predict(X)

# plt.xlabel('K')
# plt.ylabel('SSD')
#plt.plot(k, ssd)
# plt.figure(2)
plt.xlabel('K')
plt.ylabel('SC')
plt.plot(k, sc, marker='o')

optimal_n_cluster = np.argmax(sc)

print("optimal number of clusters: ", optimal_n_cluster)


# plt.scatter(X[:,0], X[:,1], c = y_kmeans, s=50, cmap='viridis')


# %%
"""
3. Visualise the data using the number of clusters and a suitable projection or low-dimentional enbedding.

"""

# %%
optimal = sorted(sc)[-5:]
plt.subplots_adjust(top=0.8, bottom=0.01, hspace=0.6, wspace=0.8)

access = map(lambda item: sc.index(item), optimal)

n_best_k = list(access)[::-1]

print(n_best_k)


# %%
pca = PCA()
pca.fit(X)

pca = PCA(n_components=20)
df_pca = pca.fit_transform(X)


# %%
figure, axs = plt.subplots(3, 2)
plt.subplots_adjust(top=0.8, bottom=0.01, hspace=0.6, wspace=0.8)

for i, k in enumerate(n_best_k):
    x = int(i/2)
    y = i % 2

    kmeans = KMeans(n_clusters=k, init='k-means++')
    Ypreds = kmeans.fit_predict(df_pca)

    axs[x, y].title.set_text(f'K = {k}')
    axs[x, y].scatter(df_pca[Ypreds == 0, 0],
                      df_pca[Ypreds == 0, 1], s=10, c='red')
    axs[x, y].scatter(df_pca[Ypreds == 1, 0],
                      df_pca[Ypreds == 1, 1], s=10, c='blue')

plt.show()
