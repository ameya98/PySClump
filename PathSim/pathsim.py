from itertools import product
from collections import defaultdict
import numpy as np

class PathSim:
    def __init__(self, type_lists, incidence_matrices):
        self.node_types = type_lists
        self.node_type_indices = {node: index for type_list in type_lists.values() for index, node in enumerate(type_list)}
        self.incidence_matrices = incidence_matrices
        self.fill_incidence_matrices()
        self.similarity_matrices = {}
    
    # Transpose for the reversed incidence type.
    def fill_incidence_matrices(self):
        initial_matrices = self.incidence_matrices.copy()
        for incidence_type in initial_matrices.keys():
            self.incidence_matrices[incidence_type[::-1]] = self.incidence_matrices[incidence_type].T

    # Find similarities between node1 and node2.
    def pathsim(self, node1, node2, metapath):
                    
        if metapath not in self.similarity_matrices:
            self.compute_similarity_matrix(metapath)
        
        try:
            return self.similarity_matrices[metapath][self.node_type_indices[node1]][self.node_type_indices[node2]]
        except ValueError:
            raise ValueError('Invalid node types for given metapath.')
        
    # Computes the similarity matrix by iterating over all pairs of nodes.
    def compute_similarity_matrix(self, metapath):

        # Compute metapath matrix.
        metapath_matrix = self.compute_metapath_matrix(metapath)

        # Fill in similarity matrix entries now.
        num_nodes = len(self.node_types[metapath[0]])
        similarity_matrix = np.eye(num_nodes, num_nodes)

        for (index1, index2) in product(range(num_nodes), range(num_nodes)):
            if index1 != index2:
                num_paths_12 = metapath_matrix[index1][index2]
                num_paths_11 = metapath_matrix[index1][index1]
                num_paths_22 = metapath_matrix[index2][index2]

                if (num_paths_11 + num_paths_22) > 0:
                    similarity_matrix[index1][index2] = (2 * num_paths_12)/(num_paths_11 + num_paths_22)
                else:
                    similarity_matrix[index1][index2] = 0
        
        self.similarity_matrices[metapath] = similarity_matrix
        return similarity_matrix

    # Computes the number of paths via this metapath.
    def compute_metapath_matrix(self, metapath):

        # We only support symmetric metapaths, for now.
        if not PathSim.symmetric(metapath):
            raise ValueError('Only symmetric metapaths supported.')
            
        curr_matrix = np.eye(len(self.node_types[metapath[0]]))
        metapath_length = len(metapath)

        for node_type, next_node_type in zip(metapath[:metapath_length//2], metapath[1:metapath_length//2 + 1]):
            
            incidence_type = node_type + next_node_type

            if incidence_type not in self.incidence_matrices.keys():
                raise ValueError('Invalid incidence %s.' % incidence_type)

            curr_matrix = np.matmul(curr_matrix, self.incidence_matrices[incidence_type])

        return np.matmul(curr_matrix, curr_matrix.T)

    # Check if metapath is symmetric. 
    @staticmethod
    def symmetric(metapath):
        metapath_length = len(metapath)
        for index, node_type in enumerate(metapath[:metapath_length//2]):
        
            other_index = metapath_length - index - 1
            other_node_type = metapath[other_index]

            if other_node_type != node_type:
                print(other_node_type, node_type)
                return False

        return True
        
    

        



