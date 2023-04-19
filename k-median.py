import numpy as np
from sklearn.cluster import KMeans

# define the 4 points
points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# define the initial centers
initial_centers = np.array([[2, 3], [6, 7]])

# use k-medians clustering with L1-norm distance metric to cluster the points into 2 clusters, starting from the initial centers
kmedians = KMeans(n_clusters=2, algorithm='median', metric='manhattan', init=initial_centers)
kmedians.fit(points)

# print the centroids of the two clusters
print(kmedians.cluster_centers_)
