from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


#read in the feature call column, split all of them by delimiter 
def generate_perturbation_matrix(
    adata,
    perturbation_col = 'feature_call',
    delim = '|',
    feature_list = None,
    sparse = True,
    ):
    print(adata)

    #if there is no feature list, split all the features in the column and build one
    if feature_list is None:
        #get all the features but not nan
        labels = adata.obs[perturbation_col].dropna()
        feature_list = labels.str.split(delim).explode().unique()

    #create a matrix of zeros with the shape of the number of cells and number of features
    perturbation_matrix = np.zeros((adata.shape[0], len(feature_list)))
    #build dicitonary mapping feature to columns index
    feature_dict = dict(zip(feature_list, range(len(feature_list))))

    #for each cell, split the feature call column by the delimiter and add 1 to the index of the feature in the feature list
    counter = 0
    for i, cell in enumerate(adata.obs[perturbation_col].str.split(delim)):
        try:
            perturbation_matrix[i, [feature_dict[feature] for feature in cell]] = 1
        except:
            counter += 1

    #ensure perturbation matrix is in the same order as adata.X
    # using feature_Dict
    #get the order of the features in the perturbation matrix

    # #split and append all
    # #put perturbation matrix in same order as adata.X
    # if inplace:
    #     adata.layers['perturbations'] = csr_matrix(perturbation_matrix)

    #print num null 

    if sparse:
        return csr_matrix(perturbation_matrix)
    else: 
        return pd.DataFrame(perturbation_matrix, index=adata.obs.index, columns=feature_list)