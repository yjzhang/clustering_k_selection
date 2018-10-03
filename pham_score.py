"""
Using Pham's method to determine optimal cluster number
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

def calculate_pham(data, clustering, km, S_k1=0.0):
    """
    See: https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/

    http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
    """
    Nd = data.shape[1]
    k_ = len(set(clustering))
    S = km.inertia_
    def calc_a(k):
        if k == 2:
            a = 1.0 - 3.0/(4.0*Nd)
        else:
            a = calc_a(k-1) + (1.0 - calc_a(k-1))/6.0
        return float(a)
    if k_==1:
        return 1, S
    elif S_k1 == 0:
        return 1, S
    else:
        a = calc_a(k_)
        print(S, S_k1, a)
        f_k = S/(a*S_k1)
        return f_k, S


def run_pham_k_selection(data, k_min=1, k_max=15):
    """
    Runs Pham's f(k) for all k from k_min to k_max.
    """
    if k_min == k_max:
        return k_min
    s_vals = [0.0]
    f_vals = []
    min_k = 0
    min_bic = np.inf
    for k in range(k_min, k_max):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        f, s = calculate_pham(data, clusters, km, s_vals[-1])
        s_vals.append(s)
        f_vals.append(f)
        if f < min_bic:
            min_k = k
            min_bic = f
    return min_k, f_vals


if __name__ == '__main__':
    import scipy.io

    data_mat = scipy.io.loadmat('../data/10x_pooled_400.mat')
    data = data_mat['data']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_pham_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_2 = scipy.io.loadmat('../../uncurl_python/data/SCDE_k2_sup.mat')
    data = data_mat_2['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_pham_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_mat_3 = scipy.io.loadmat('../../uncurl_python/data/GSE60361_dat.mat')
    data = data_mat_3['Dat']
    data_tsvd = preproc_data(data)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_pham_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_4 = scipy.io.mmread('../../split_seq_analysis/spinal_cord.mtx.gz')
    import scipy.sparse
    data_4 = scipy.sparse.csc_matrix(data_4)
    data_tsvd = preproc_data(data_4)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_pham_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)

    data_5 = scipy.io.mmread('../../uncurl_test_datasets/10x_pure_pooled/data_8000_cells.mtx.gz')
    data_5 = scipy.sparse.csc_matrix(data_5)
    data_tsvd = preproc_data(data_5)
    #max_k, bic_vals = run_bic_k_selection(data_tsvd)
    max_k, bic_vals = run_pham_k_selection(data_tsvd)
    print(max_k)
    print(bic_vals)


