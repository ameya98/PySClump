# SClump
A Python Implementation of 'Spectral Clustering in Heterogeneous Information Networks' by Xiang Li, Ben Kao, Zhaochun Ren, Dawei Yin, at AAAI 2019.

## PathSim
We use PathSim as a similarity metric between pairs of nodes.
```
from pathsim import PathSim
import numpy as np

type_lists = {
    'A': ['Mike', 'Jim', 'Mary', 'Bob', 'Ann'],
    'C': ['SIGMOD', 'VLDB', 'ICDE', 'KDD'],
    'V': ['Pasadena', 'Guwahati', 'Bangalore']
}

incidence_matrices = { 
   'AC': np.array([[2, 1, 0, 0], [50, 20, 0, 0], [2, 0, 1, 0], [2, 1, 0, 0], [0, 0, 1, 1]]),
   'VC': np.array([[3, 1, 1, 1], [1, 0, 0, 0], [2, 1, 0, 1]])
}

# Create PathSim object.
ps = PathSim(type_lists, incidence_matrices)

# Get the similarity between two authors (indicated by type 'A').
ps.pathsim('Mike', 'Jim', metapath='ACA')

# Get the similarity matrix M for the metapath.
ps.compute_similarity_matrix(metapath='ACVCA')
```
