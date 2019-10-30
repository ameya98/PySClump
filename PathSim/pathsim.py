from itertools import product
from collections import defaultdict

class PathSim:
    def __init__(self, network):
        self.network = network
        self.nodes_by_type = defaultdict(list)
        self.find_node_types()
    
    # Fill nodes by type.
    def find_node_types(self):
        for node_index, node in enumerate(self.network.nodes):
            self.nodes_by_type[node['type']].append(node_index)

    # Computes the similarity matrix by iterating over all pairs of nodes.
    def compute_similarity_matrix(self, metapath):

        # Compute metapath matrix.
        metapath_matrix = compute_metapath_matrix(metapath)

        # Fill in similarity matrix entries now.
        num_nodes = self.network.number_of_nodes()
        similarity_matrix = np.zeros((num_nodes, num_nodes))

        for (index1, index2) in product(range(num_nodes), range(num_nodes)):
            if index1 != index2:
                num_paths_12 = metapath_matrix[index1][index2]
                num_paths_11 = metapath_matrix[index1][index1]
                num_paths_22 = metapath_matrix[index2][index2]

                similarity_matrix[index1][index2] = (2 * num_paths_12)/(num_paths_11 + num_paths_22)
        
        return similarity_matrix

    # Computes the number of paths via this metapath.
    def compute_metapath_matrix(self, metapath):

        # We only support symmetric metapaths, for now.
        if not PathSim.symmetric(metapath):
            raise ValueError('Only symmetric metapaths supported.')
            
        curr_matrix = np.eye(self.network.number_of_nodes())
        metapath_length = len(metapath)

        for node_type, next_node_type in zip(metapath[:metapath_length/2], metapath[:metapath_length/2]):
            for node, next_node in product(self.nodes_by_type[node_type], self.nodes_by_type[node_type]):
                curr_matrix[node][next_node] += self.incidence_matrix[node_type + next_node_type][node][next_node]

        return curr_matrix

    # Check if metapath is symmetric. 
    @staticmethod
    def symmetric(metapath):
        metapath_length = len(metapath)
        for index, node_type in enumerate(metapath[:metapath_length/2]):
        
            other_index = metapath_length - index - 1
            other_node_type = metapath_length

            if other_node_type != node_type:
                return False

        return metapath_length
        

        
        



