"""
Using gap score to determine optimal cluster number
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

def calculate_bounding_box(data):
    """
    Returns a 2 x m array indicating the min and max along each
    dimension.
    """
    mins = data.min(0)
    maxes = data.max(0)
    return mins, maxes

def calculate_gap(data, clustering, km, B=50):
    """
    See: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/

    https://web.stanford.edu/~hastie/Papers/gap.pdf

    Returns two results: the gap score, and s_k.
    """
    k = len(set(clustering))
    Wk = km.inertia_
    mins, maxes = calculate_bounding_box(data)
    Wk_est = []
    for i in range(B):
        data_sample = (maxes-mins)*np.random.random(data.shape) + mins
        km_b = KMeans(k)
        km_b.fit_predict(data_sample)
        Wk_est.append(km_b.inertia_)
    Wk_est = np.log(np.array(Wk_est))
    Wk_mean = np.mean(Wk_est)
    Wk_std = np.std(Wk_est)
    gap = Wk_mean - np.log(Wk)
    sk = np.sqrt(1 + 1.0/B)*Wk_std
    return gap, sk


def run_gap_k_selection(data, k_min=1, k_max=15, B=50,
        skip=1):
    """
    Runs gap score for all k from k_min to k_max.
    """
    if k_min == k_max:
        return k_min
    gap_vals = []
    sk_vals = []
    k_range = list(range(k_min, k_max, skip))
    min_k = 0
    min_i = 0
    for i, k in enumerate(k_range):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        gap, sk = calculate_gap(data, clusters, km, B=B)
        if len(gap_vals) > 1:
            if gap_vals[-1] >= gap - (skip+1)*sk:
                min_i = i
                min_k = k_range[i-1]
                break
                #return k_range[-1], gap_vals, sk_vals
        gap_vals.append(gap)
        sk_vals.append(sk)
    if min_k == 0:
        min_k = k_max
    if skip == 1:
        return min_k, gap_vals, sk_vals
    gap_vals = []
    sk_vals = []
    for k in range(min_k - skip, min_k + skip):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        gap, sk = calculate_gap(data, clusters, km, B=B)
        if len(gap_vals) > 1:
            if gap_vals[-1] >= gap - sk:
                min_k = k-1
                return min_k, gap_vals, sk_vals
        gap_vals.append(gap)
        sk_vals.append(sk)
    return k, gap_vals, sk_vals


if __name__ == '__main__':
    import scipy.io
    import time

    data_mat = scipy.io.loadmat('../data/10x_pooled_400.mat')
    data = data_mat['data']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    t0 = time.time()
    max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, 
            k_min=1, k_max=50, skip=5, B=5)
    print(max_k)
    print(gap_vals)
    print(sk_vals)
    print('time: ' + str(time.time() - t0))

    data_mat_2 = scipy.io.loadmat('../../uncurl_python/data/SCDE_k2_sup.mat')
    data = data_mat_2['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    #max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, B=5)
    t0 = time.time()
    max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, 
            k_min=1, k_max=50, skip=5, B=5)
 
    print(max_k)
    print(gap_vals)
    print(sk_vals)
    print('time: ' + str(time.time() - t0))

    data_mat_3 = scipy.io.loadmat('../../uncurl_python/data/GSE60361_dat.mat')
    data = data_mat_3['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    #max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, B=5)
    t0 = time.time()
    max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, 
            k_min=1, k_max=50, skip=5, B=5)
 
    print(max_k)
    print(gap_vals)
    print(sk_vals)
    print('time: ' + str(time.time() - t0))

    data_4 = scipy.io.mmread('../../split_seq_analysis/spinal_cord.mtx.gz')
    import scipy.sparse
    data_4 = scipy.sparse.csc_matrix(data_4)
    data_tsvd = preproc_data(data_4)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    #max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, B=5)
    t0 = time.time()
    max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, 
            k_min=1, k_max=50, skip=5, B=5)
 
    print(max_k)
    print(gap_vals)
    print(sk_vals)
    print('time: ' + str(time.time() - t0))

    data_5 = scipy.io.mmread('../../uncurl_test_datasets/10x_pure_pooled/data_8000_cells.mtx.gz')
    data_5 = scipy.sparse.csc_matrix(data_5)
    data_tsvd = preproc_data(data_5)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    #max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, B=5)
    t0 = time.time()
    max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd, 
            k_min=1, k_max=50, skip=5, B=5)
 
    print(max_k)
    print(gap_vals)
    print(sk_vals)
    print('time: ' + str(time.time() - t0))


