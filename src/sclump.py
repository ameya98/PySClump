"""
A Python Implementation of 'Spectral Clustering in Heterogeneous Information Networks' by Xiang Li, Ben Kao, Zhaochun Ren, Dawei Yin, at AAAI 2019.
"""

# External dependencies.
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import minimize

# Internal dependencies.
from utils import eigenvalues, eigenvectors, normalized_laplacian, distance_matrix, best_simplex_projection

class SClump:
    def __init__(self, similarity_matrices, num_clusters, random_seed=0):
        
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.num_nodes = None
        self.similarity_matrices = []
        self.metapath_index = {}

        # Loss function coefficients. We'll fill this in later.
        self.alpha = None
        self.beta = None
        self.gamma = None

        for index, (metapath, matrix) in enumerate(similarity_matrices.items()):
            if self.num_nodes is None:
                self.num_nodes = matrix.shape[0]
            
            if matrix.shape != (self.num_nodes, self.num_nodes):
                raise ValueError('Invalid shape of similarity matrix.')
            
            # Normalize so that row sums are 1.
            row_normalized_matrix = matrix/matrix.sum(axis=1, keepdims=True)
            
            self.similarity_matrices.append(row_normalized_matrix)
            self.metapath_index[metapath] = index

        self.similarity_matrices = np.array(self.similarity_matrices)
        self.num_metapaths = len(similarity_matrices)
        

    def run(self, verbose=False):
        """
        Returns a tuple of:
        * labels: The predicted cluster labels for each node.
        * similarity: The learnt similarity matrix, from the optimization procedure.
        """
        similarity_matrix = self.optimize(verbose=verbose)
        labels = self.cluster(similarity_matrix)

        return labels, similarity_matrix
    
    
    def cluster(self, feature_matrix):
        """
        Cluster row-wise entries in the feature matrix. Currently, k-means clustering is used.
        TODO: Add spectral rotation as an alternative to k-means.
        """
        return KMeans(self.num_clusters, n_init=10, random_state=self.random_seed).fit_predict(feature_matrix)


    def optimize(self, num_iterations=20, alpha=0.5, beta=10, gamma=0.01, verbose=False):
        """
        Learn weights to optimize the similarity matrix.
       
        Coefficients in the loss function:
        * alpha: for the Frobenius norm of S, 
        * beta: for the L2-norm of lambda, 
        * gamma: for the trace of LS.
        """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Weights over similarity matrices.
        lambdas = np.ones(self.num_metapaths)/self.num_metapaths
        W = np.tensordot(lambdas, self.similarity_matrices, axes=[[0], [0]])
        
        # Initialize.
        S = W

        # Iterate.
        for iteration in range(num_iterations):
            
            if verbose:
                loss = np.trace(np.matmul((S - W).T, (S - W))) 
                loss += self.alpha * np.trace(np.matmul(S.T, S))
                loss += self.beta * np.dot(lambdas, lambdas)
                loss += self.gamma * np.sum(eigenvalues(normalized_laplacian(S), num=self.num_clusters))

                print('Iteration %d: Loss = %0.3f' % (iteration, loss))

            # Update F.
            F = self.optimize_F(S)

            # Update S.
            S = self.optimize_S(W, F)
            
            # Update lambdas.
            lambdas = self.optimize_lambdas(S, lambdas)

            # Recompute W.
            W = np.tensordot(lambdas, self.similarity_matrices, axes=[[0], [0]])

        return S


    # Optimize F, keeping S fixed.
    def optimize_F(self, S):
        LS = normalized_laplacian(S)    
        return eigenvectors(LS, num=self.num_clusters)


    # Optimize S, keeping W and F fixed.
    def optimize_S(self, W, F):
        Q = distance_matrix(F, metric='euclidean')
        P = (2*W - self.gamma*Q)/(2 + 2*self.alpha)

        S = np.zeros((self.num_nodes, self.num_nodes)) 
        for index in range(S.shape[0]):
            S[index] = best_simplex_projection(P[index])
        
        return S
    
    # Optimize lambdas, keeping S fixed.
    def optimize_lambdas(self, S, init_lambdas):

        # The objective function, quadratic in lambda.
        def objective(lambdas):
            W = np.tensordot(lambdas, self.similarity_matrices, axes=[[0], [0]])
            value = np.trace(np.matmul(W.T, W))
            value -= 2 * np.trace(np.matmul(S.T, W))
            value += self.beta * np.dot(lambdas, lambdas)
            return value
    
        # Constraints are non-negativity and 1-summation.
        # Non-negativity is handled by the bounds() function. 
        def constraints():
            def sum_one(lambdas):
                return np.sum(lambdas) - 1

            return  {
                'type': 'eq',
                'fun': sum_one,
            }

        # Handle non-negativity
        def bounds(init_lambdas):
            return [(0, 1) for init_lambda in init_lambdas]

        return minimize(objective, init_lambdas, method='SLSQP', constraints=constraints(), bounds=bounds(init_lambdas)).x

