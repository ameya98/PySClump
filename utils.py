"""
Utility functions. Mostly math stuff.
"""

import numpy as np
import scipy as sc


# Returns the top eigenvectors (according to increasing eigenvalues) of a symmetric matrix.
def eigenvectors(matrix, num_eigenvectors=1):
    return sc.linalg.eigh(matrix, eigvals=(0, num_eigenvectors - 1))[1]


# Returns the normalized Laplacian for a given similarity matrix.
def normalized_laplacian(similarity_matrix):
    A = similarity_matrix
    R = np.sum(A, axis=1)
    R_minus_half = 1/sqrt(R)
    D_minus_half = np.diag(R_minus_half)
    I = np.eye(A.shape[0])
    return I - D_minus_half * A * D_minus_half


# Returns the unnormalized Laplacian for a given similarity matrix.
def unnormalized_laplacian(similarity_matrix):
    A = similarity_matrix
    R = np.sum(A, axis=1)
    D = np.diag(R)
    return D - A


# Returns the pairwise distance between the row-wise vectors (representing samples).
def distance_matrix(matrix, metric):
    scipy.spatial.distance.pdist(matrix, metric=metric)


# Solves the sparse simplex reconstruction problem for a vector, v.
# Follows notation given in 'A New Simplex Sparse Learning Model to Measure Data Similarity for Clustering', IJCAI 2015.
def best_simplex_projection(vector, num_iterations=100):

    # Compute quantities.
    v = vector
    n = v.size()
    u = v - np.mean(v) + np.ones(n)/n

    if np.min(u) < 0:
        
        func = 1
        multiplier = 0
        iteration = 0

        while np.abs(f) > 1e-10 and iteration < num_iterations:
            alpha_uncut = u - multiplier
            indices = (alpha_uncut > 0)
            
            # Compute function value and derivative.
            func_deriv = -np.sum(indices)
            func = np.sum(alpha_uncut[indices]) - 1
            
            # Newton's rule update.
            multiplier = multiplier - func/func_deriv    

        alpha = np.max(alpha, 0)

    else:
        alpha = u
    
    return alpha
