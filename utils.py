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