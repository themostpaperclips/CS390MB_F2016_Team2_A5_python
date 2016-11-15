"""
Unfortunately, the starter code makes unit tests impossible.
This is a hacky way to run some basic unit tests and make sure everything works.
"""

import socket
import sys
import json
import numpy as np
from sklearn.cluster import KMeans, MeanShift
import unittest

def cluster(latitudes, longitudes, algorithm, *args):
    """
    This function is identical to cluster in location_clustering.py
    but it returns the labels rather than sending them. Use for testing
    """

    X = np.column_stack((np.array(latitudes), np.array(longitudes)))

    if algorithm == "k_means":
        kmeans = KMeans(n_clusters=args[0]).fit(X)
        return kmeans.labels_
    else:
        meanShift = MeanShift().fit(X)
        return meanShift.labels_


class TestClustering(unittest.TestCase):
    def test_kmeans(self):
        labels = cluster([0, 100, 1, 101], [0, 100, 1, 101], "k_means", 2)
        self.assertTrue(np.array_equal(labels, [0, 1, 0, 1]) or np.array_equal(labels, [1, 0, 1, 0]))

    def test_meanshift(self):
        labels = cluster([0, 100, 1, 101, 2, 3, 99, 103], [0, 100, 1, 101, 2, 3, 99, 103], "mean_shift")
        self.assertTrue(np.array_equal(labels, [2, 0, 2, 0, 1, 1, 0, 3]))

if __name__ == '__main__':
    unittest.main()
