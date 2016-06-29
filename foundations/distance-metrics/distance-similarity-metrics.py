# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

# data set https://archive.ics.uci.edu/ml/datasets/Iris


# EUCLIDEAN DISTANCE

# Euclidean distance is the most common distance metric and is simply the
# length of the line segment connecting two vectors in a vector space. Also
# known as the "ruler distance," an intuitive way of thinking about Euclidean
# distance (at least in 2-dimensional space) is to take a ruler and draw a
# line between two points on paper; the length of the line drawn is the
# Euclidean distance between the two points.


# Euclidean distance between two vectors
np.random.seed(1000)
x = [np.random.rand(1, 50) for _ in range(50)]
y = np.zeros(50)
dist = np.linalg.norm(x - y)


# x is a list of random vectors, each of which has 50 features.
# How far is the closest point to the origin, (0,0)(0,0)?
closest_point = -1

for el in x:
    dist = np.linalg.norm(el - y)
    if closest_point >= 0 and closest_point > dist or closest_point == -1:
        closest_point = dist

print closest_point


# alternatively
d = [np.linalg.norm(x0 - y) for x0 in x]
print min(d)


# COSINE SIMILARITY

# Cosine similarity measures the similarity between two vectors
# (observations) as the cosine of the angle between them. We can
# derive this measure using the dot product of vectors A and B:
# a . b = |a| . |b| . cos(theta)

# The cosine similarity consists of two parts: a direction and a magnitude.
# The direction (sign) of the similarity score indicates whether the two
# objects are similar or dissimilar. The magnitude measures the strength
# of the relationship between the two objects.

# We can compute this quite easily for vectors xx and yy using SciPy,
# by modifying the cosine distance function:
# 1 + scipy.spatial.distance.cosine(x, y)

# We add "1" for rescaling purposes, since SciPy's function returns the
# distance (by computing 1 - cosine similarity) rather than similarity. To
# perform the underlying computation yourself, you can use the following code:
np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# JACCARD INDEX AND JACCARD DISTANCE

# The Jaccard Index, also known as the Jaccard Similarity Coefficient,
# is designed to measure the proportion of unique data points that exist
# in two sets A and B. This can be expressed as:
# J(A, B) = | A intersection B | / | A union B |

# We can also construct a measure of the dissimilarity between the sets
# AA and BB, known as the Jaccard Distance, which according to
# De Morgan's Law, is 1−J(A,B).

# The Jaccard index can be computed using the following lines of code:
def jaccardIndex(A, B):
    A = set(A)
    B = set(B)
    num = len(A.intersection(B))
    return (float(num) / (len(A) + len(B) - num))


# The Jaccard distance can be computed using SciPy's spatial.distance module:
sp.spatial.distance.jaccard(x, y)


# The Jaccard distance has its origins in biology, where it was first
# used as a measure for comparing the diversity among species, and is
# most often used as a simple similarity matching metric. An obvious
# shortcoming of the metric is that it fails to capture the relative
# frequency and weights (i.e., importance) of observations in the sets.


# MAHALANOBIS DISTANCE

# The Mahalanobis distance is a generalization of the Euclidean distance,
# which addresses differences in the distributions of feature vectors, as
# well as correlations between features.

# Given two vectors, XX and YY, and letting the quantity dd denote
# the Mahalanobis distance, we can express the metric as follows:
# d(X, Y) = sqrt((X - Y).T * Σ^−1 * (X - Y ))

# where Σ−1 is the inverse of the covariance matrix of the data set.
# The covariance matrix represents the covariances between each pair of
# features. One way to think about the covariance matrix is as a
# mathematical description of how spread out the data is in different
# dimensions. The inverse of the covariance matrix is used to transform
# the data so that each feature becomes uncorrelated with all other
# features and all transformed features have the same amount of variance,
# which eliminates the scaling issues present in Euclidean distance.
data = np.genfromtxt('Jedi.data', delimiter=',', usecols=range(4))

# What is the first element in the covariance matrix?
covariance_matrix = np.cov(data.T)[0][0]

# Now, let's compute the inverse covariance matrix. What is the first
# element in the inverse covariance matrix?
inverse_cov_matrix = np.linalg.inv(np.cov(data.T))

# Compute the Mahalanobis distance between the first and second Jedi
# in the data matrix. Round your answer to four decimal places.
data = np.genfromtxt('Jedi.data', delimiter=',', usecols=range(4))
invCov = np.linalg.inv(np.cov(data, rowvar=0))
x = data[0]
y = data[1]
np.sqrt(np.dot(np.dot((x - y).T, invCov), (x - y)))

# LEVENSHTEIN DISTANCE

# This problem can be solved with an algorithm known as the Levenshtein
# distance, which has numerous applications in text processing, computer
# science, and genetics. The Levenshtein distance gives us a way of
# computing the edit distance between two strings (or any two sequences
# of characters), which is the minimum number of edit operations
# (addition, deletion, substitution of characters) needed to transform
# "wuieksort" into another string, such as "quicksort."

# The Levenshtein distance is closely related to the "edit distance."
# Where the Levenshtein distance assigns a uniform cost of 1 for any operation,
# edit distance assigns an arbitrary cost to addition, deletion, and
# substitution operations. This allows a user to customize the priorities
# of certain types of edits.

# IMPLEMENTATION: DYNAMIC PROGRAMMING

# In cases where we do need to compute the Levenshtein distance,
# we can use a bottom-up Dynamic Programming approach. With this approach,
# we compute the edit distance of strings A and B by first computing the
# edit distances of their respective substrings.
