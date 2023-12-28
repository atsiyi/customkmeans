import numpy as np
import pandas as pd
from numpy.random import uniform
import random
import re
import requests
import json

def get_distancematrix(df):
    # string prep for location pair
    location_list = []
    location_pair = zip(list(df['longitude']), list(df['latitude']))
    for f, b in zip(df['longitude'], df['latitude']):
        location_list.append(f)
        location_list.append(b)
    locpair = ';'.join(map(str, location_pair))
    locpair=re.sub("[(|) ]","",locpair)
    # set string to call OSRM API
    getdistance_url = {'OSRM_API_ADDRESS'} + locpair + '?annotations=distance'
    text = requests.get(getdistance_url)
    jsontext = json.loads(text.content)
    df_distance = pd.DataFrame(jsontext['distances'])/1000 # get distance in km
    return(df_distance)

def driving_distance(point, data):
    points, drivingdist = [],[]
    points.append(point)
    for x in data:
        points.append(x)
    dist_points = get_distancematrix(pd.DataFrame(list(points), columns=['longitude', 'latitude']))
    for i,point in enumerate(points):
        dist = dist_points.loc[0,i]
        drivingdist.append(dist)
    drivingdist.pop(0)
    return(drivingdist)

class KMeans:

    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train, initcentroid):
        # initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # pick a random point from train data for first centroid
        self.centroids = initcentroid #[random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            # calculate distances from points to the centroids
            dists = np.sum([driving_distance(centroid, X_train) for centroid in self.centroids], axis=0) 
            # dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0) #for euclidean
            # normalize the distances
            dists /= np.sum(dists)
            # choose remaining points based on their distances
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
            self.centroids += [X_train[new_centroid_idx]]

        # randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = driving_distance(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = driving_distance(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return(centroids, centroid_idxs)