"""
Using BIC to determine optimal cluster number
"""
import numpy as np
from sklearn.cluster import KMeans

def preproc_data(data):
    """
    basic data preprocessing
    """
    import uncurl
    from uncurl.preprocessing import log1p, cell_normalize
    from sklearn.decomposition import TruncatedSVD
    gene_subset = uncurl.max_variance_genes(data)
    data_subset = data[gene_subset, :]
    tsvd = TruncatedSVD(8)
    data_tsvd = tsvd.fit_transform(log1p(cell_normalize(data_subset)).T)
    return data_tsvd

def calculate_bic(data, clustering, km):
    """
    Calculates the BIC score for a clustering...
    Data likelihood is just sum[(X[cluster,:] - cluster_center)^2]

    Args:
        data (array): cells x genes
        clustering (array): 1d array of ints
    """
    n = data.shape[0]
    clusters = set(clustering)
    # reference: https://nlp.stanford.edu/IR-book/html/htmledition/cluster-cardinality-in-k-means-1.html
    return np.log(n)*data.shape[1]*len(clusters) + km.inertia_

def run_bic_k_selection(data, k_min=2, k_max=80, grid_size=5):
    """
    runs Bayes information criterion using a grid search
    kind of thing...???
    """
    if k_min == k_max:
        return k_min
    bic_vals = []
    min_k = 0
    min_bic = np.inf
    for k in range(k_min, k_max, grid_size):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        bic = calculate_bic(data, clusters, km)
        bic_vals.append(bic)
        if bic < min_bic:
            min_k = k
            min_bic = bic
    if grid_size > 1:
        for k in range(min_k - grid_size, min_k + grid_size):
            if k <= 0:
                continue
            km = KMeans(k)
            clusters = km.fit_predict(data)
            bic = calculate_bic(data, clusters, km)
            if bic < min_bic:
                min_k = k
                min_bic = bic
    return min_k, bic_vals

def run_silhouette_k_selection(data, k_min=2, k_max=80, grid_size=5):
    """
    tries to find the k with the max silhouette score using a grid search
    kind of thing...???
    """
    from sklearn.metrics import silhouette_score
    if k_min == k_max:
        return k_min
    s_scores = []
    max_k = 0
    max_s = -1
    for k in range(k_min, k_max, grid_size):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        s = silhouette_score(data, clusters)
        s_scores.append(s)
        if s > max_s:
            max_k = k
            max_s = s
    if grid_size > 1:
        for k in range(max_k - grid_size, max_k + grid_size):
            if k <= 1:
                continue
            km = KMeans(k)
            clusters = km.fit_predict(data)
            s = silhouette_score(data, clusters)
            if s > max_k:
                max_k = k
                max_s = s
    return max_k, s_scores




if __name__ == '__main__':
    import scipy.io

    data_mat = scipy.io.loadmat('../data/10x_pooled_400.mat')
    data = data_mat['data']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_silhouette_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_2 = scipy.io.loadmat('../../uncurl_python/data/SCDE_k2_sup.mat')
    data = data_mat_2['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_silhouette_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_3 = scipy.io.loadmat('../../uncurl_python/data/GSE60361_dat.mat')
    data = data_mat_3['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_silhouette_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_4 = scipy.io.mmread('../../split_seq_analysis/spinal_cord.mtx.gz')
    import scipy.sparse
    data_4 = scipy.sparse.csc_matrix(data_4)
    data_tsvd = preproc_data(data_4)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_silhouette_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_5 = scipy.io.mmread('../../uncurl_test_datasets/10x_pure_pooled/data_8000_cells.mtx.gz')
    data_5 = scipy.sparse.csc_matrix(data_5)
    data_tsvd = preproc_data(data_5)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_silhouette_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)


