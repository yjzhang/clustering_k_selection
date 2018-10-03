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
    Data likelihood is just (X[cluster,:] - cluster_center)^2

    Args:
        data (array): cells x genes
        clustering (array): 1d array of ints
    """
    n = data.shape[0]
    clusters = set(clustering)
    # calculate cluster centers
    return np.log(n)*data.shape[1]*len(clusters) + km.inertia_
    #ll = 0
    #cluster_centers = np.zeros((len(clusters), data.shape[1]))
    #pi2 = 0.5*np.log(2*np.pi)
    #for c in clusters:
    #    cluster_centers[c,:] = np.mean(data[clustering==c,:])
    #    # calculate log-likelihood
    #    # variance for each gene/feature
    #    var = data[clustering==c,:].var(0)
    #    if np.isinf(var).any() or np.isnan(var).any() or (var==0).any():
    #        return np.inf
    #    # sum of differences for each feature
    #    d2 = ((data[clustering==c,:] - cluster_centers[c,:])**2)
    #    lls = -pi2 - np.log(var) - d2/(2*var**2)
    #    ll += lls.sum()
    #return np.log(n)*len(clusters)*data.shape[1] - 2*ll


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



if __name__ == '__main__':
    import scipy.io
    import uncurl
    from uncurl.preprocessing import log1p, cell_normalize
    from sklearn.decomposition import TruncatedSVD

    data_mat = scipy.io.loadmat('../data/10x_pooled_400.mat')
    data = data_mat['data']
    data_tsvd = preproc_data(data)
    max_k, bic_vals = run_bic_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_2 = scipy.io.loadmat('../../uncurl_python/data/SCDE_k2_sup.mat')
    data = data_mat_2['Dat']
    data_tsvd = preproc_data(data)
    max_k, bic_vals = run_bic_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_3 = scipy.io.loadmat('../../uncurl_python/data/GSE60361_dat.mat')
    data = data_mat_3['Dat']
    data_tsvd = preproc_data(data)
    max_k, bic_vals = run_bic_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)


