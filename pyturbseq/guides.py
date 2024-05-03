##########################################################################
# 
# Functions for QC of sgRNA (guide) libraries
#
##########################################################################
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np

def hamming_dist(a, b):
    return distance.hamming(list(a), list(b))

def hamming_dist_matrix(ref):
    """
    This method take a list of strings, assumed to all be the same length, and pairwise haming distance
    """
    dists = np.zeros((len(ref), len(ref)))

    #get triangle upper indices
    triu_indices = np.triu_indices(len(ref), k=0)

    # iterate over trianlge indices
    for i, j in tqdm(zip(*triu_indices)):
        d = hamming_dist(ref[i], ref[j])
        dists[i, j] = d
        dists[j, i] = d

    return dists