##########################################################################
# 
# Functions for QC of sgRNA (guide) libraries
#
##########################################################################
from typing import Iterable
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np

def hamming_dist(a: str, b: str) -> float:
    """Compute the normalized Hamming distance between two strings.
    
    Args:
        a: First string.
        b: Second string.
    Returns:
        Normalized Hamming distance between the two strings.
    """
    return distance.hamming(list(a), list(b))

def hamming_dist_matrix(ref: Iterable[str]) -> np.ndarray:
    """Compute pairwise Hamming distance matrix for a collection of strings.

    Args:
        ref: Iterable of strings, all assumed to be equal length.
    Returns:
        A square NumPy array of pairwise Hamming distances.
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
