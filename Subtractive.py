import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn.metrics.pairwise
import operator
from sklearn.cluster import KMeans
import scipy.spatial.distance as distance
from scipy.spatial.distance import cdist

path = "C:/Users/Leo/Documents/Pre blackboard/Semester 1 17-18/Data Prog with Python/Project 2/data.csv"
df = pd.read_csv(path, header=None, names = ['X1', 'X2', 'X3'])
print(df.head())
print(df.describe())

#Visualise data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X1'], df['X2'], df['X3'], s=8,c='blue', marker='o')
ax.set_title('Original data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

#Step 1 - Normalise
df=(df-df.min())/(df.max()-df.min())
print(df.describe())

#Step 2 - Compute Pi
ra = .8
rb = 1.25*ra
Eup = 0.5
Edown = 0.15

dist = sklearn.metrics.pairwise_distances(df, metric='l2')
Pi = np.sum(np.exp(-((dist**2)/((ra/2)**2))), axis=1)

#Step 3 - Select max of Pi
accepted_potentials = np.array([])
index, potential_1 = max(enumerate(Pi), key=operator.itemgetter(1))
X_1 = df.iloc[index,:]
accepted_centers = np.array([X_1])
accepted_potentials = np.append(accepted_potentials, potential_1)
accepted_indices = np.array(index)

#Step 4 - Update potential values
diff2 = dist[index,:]
Pi = Pi - potential_1 * (np.exp(-((diff2**2)/((rb/2)**2))))

#Step 5 - Select highest potential Pk* from updated Pi
index_k, potential_k = max(enumerate(Pi), key=operator.itemgetter(1))
while (potential_k/potential_1) >= Edown:
    index_k, potential_k = max(enumerate(Pi), key=operator.itemgetter(1))
    X_k = df.iloc[index_k,:]

    #Step 6 - Accept/reject candidate from step 5
    potential_centers = np.vstack([accepted_centers, X_k])

    if (potential_k/potential_1) > Eup:
        accepted_centers = np.vstack([accepted_centers, X_k])
        accepted_indices = np.append(accepted_indices, index_k)
        diff_k = dist[index_k,:]
        Pi = Pi - potential_k * (np.exp(-((diff_k ** 2) / ((rb / 2) ** 2)))) #Step 7 - Compute potential for remaining data
        index_k, potential_k = max(enumerate(Pi), key=operator.itemgetter(1))
        X_k = df.iloc[index_k, :]

    elif Edown <= (potential_k/potential_1) <= Eup:
        dmin_matrix = sklearn.metrics.pairwise_distances(potential_centers, metric='l2')
        dmin = np.min(dmin_matrix[np.nonzero(dmin_matrix)])
        if ((dmin/ra) + (potential_k/potential_1)) >= 1:
            accepted_centers = np.vstack([accepted_centers, X_k])
            accepted_indices = np.append(accepted_indices, index_k)
            Pi = Pi - potential_k * (np.exp(-((diff_k ** 2) / ((rb / 2) ** 2))))
            index_k, potential_k = max(enumerate(Pi), key=operator.itemgetter(1))

        else:
            Pi = np.delete(Pi, index_k)
            index_k, potential_k = max(enumerate(Pi), key=operator.itemgetter(1))

print("Subtractive clustering centers:")
print(accepted_centers)

#Plot the clusters with centers
labels = df.apply(lambda row: np.argmin(np.linalg.norm(accepted_centers - row.values, axis=1)), axis=1)
colors = ['red', 'green', 'blue']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['X1'], df['X2'], df['X3'], c=labels, s=8, cmap=matplotlib.colors.ListedColormap(colors))
ax.scatter(accepted_centers[0,:], accepted_centers[1,:], accepted_centers[2,:], c='black', s=200, alpha=1, marker='*')
ax.set_title('Subtractive Clustering')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

#Compare with Kmeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=3125)
kmeans = kmeans.fit(df)
labels = kmeans.predict(df)
centers = kmeans.cluster_centers_
print("K-Means centers:")
print(centers)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=labels, s=8, cmap=matplotlib.colors.ListedColormap(colors))
ax.scatter(centers[0,:], centers[1,:], centers[2,:], c='black', s=200, alpha=1, marker='*')
ax.set_title('K-Means Clustering')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

#Check optimal number of clusters
distortions = []
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    kmeans.fit(df)
    distortions.append(sum(np.min(cdist(df, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method - Optimal k')
plt.show()

