# DPMM clustering with bnpy
import bnpy
import numpy as np
import scipy.io
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import uncurl
from uncurl.preprocessing import log1p, cell_normalize


def bnpy_select_clusters(data, max_cells=50000):
    """
    Args:
        data: matrix of shape genes x cells

    Returns:
        selected k based on converged Gaussian DPMM, and
            the assigned labels.
    """
    # TODO: randomly sub-select max_cells
    selected_cell_ids = list(range(data.shape[1]))
    if max_cells < data.shape[1]:
        import random
        selected_cell_ids = random.sample(selected_cell_ids, max_cells)
    data = data[:, selected_cell_ids]
    tsvd = TruncatedSVD(8)
    data_tsvd = tsvd.fit_transform(log1p(cell_normalize(data)).T)
    data_dense_bnpy = bnpy.data.XData(data_tsvd)
    trained_model, info_dict = bnpy.run(
            data_dense_bnpy, 'DPMixtureModel', 'Gauss', 'memoVB',
            #doSaveToDisk=False,
            doWriteStdOut=False,
            output_path='./temp',
            nLap=100, nTask=1, nBatch=1,
            sF=0.1, ECovMat='eye',
            K=10, initname='randexamples',
            moves='birth,merge,shuffle',
            m_startLap=5, b_startLap=2, b_Kfresh=4)
    selected_k = info_dict['K_history'][-1]
    results = trained_model.calc_local_params(data_dense_bnpy)
    cluster_labels =  results['resp'].argmax(1)
    return selected_k, cluster_labels

if __name__ == '__main__':
    import time
    # load/subset data
    data_mat = scipy.io.loadmat('../data/10x_pooled_400.mat')
    data = data_mat['data']
    gene_subset = uncurl.max_variance_genes(data)
    data_subset = data[gene_subset, :]

    # run bnpy clustering?
    true_labels = data_mat['labels'].flatten()
    t0 = time.time()
    selected_k, labels = bnpy_select_clusters(data_subset)
    print(selected_k)
    print('nmi: ' + str(nmi(true_labels, labels)))
    print('time: ' + str(time.time() - t0))

    data_mat_2 = scipy.io.loadmat('../../uncurl_python/data/SCDE_k2_sup.mat')
    data = data_mat_2['Dat']
    t0 = time.time()
    selected_k, labels = bnpy_select_clusters(data)
    true_labels = data_mat_2['Lab'].flatten()
    print(selected_k)
    print('nmi: ' + str(nmi(true_labels, labels)))
    print('time: ' + str(time.time() - t0))

    # Zeisel 7-cluster dataset
    data_mat_3 = scipy.io.loadmat('../../uncurl_python/data/GSE60361_dat.mat')
    data = data_mat_3['Dat']
    gene_subset = uncurl.max_variance_genes(data)
    data_subset = data[gene_subset, :]
    true_labels = data_mat_3['ActLabs'].flatten()
    t0 = time.time()
    selected_k, labels = bnpy_select_clusters(data)
    print(selected_k)
    print('nmi: ' + str(nmi(true_labels, labels)))
    print('time: ' + str(time.time() - t0))

    # spinal cord data
    data_4 = scipy.io.mmread('../../split_seq_analysis/spinal_cord.mtx.gz')
    import scipy.sparse
    data_4 = scipy.sparse.csc_matrix(data_4)
    gene_subset = uncurl.max_variance_genes(data_4)
    data_subset = data_4[gene_subset, :]
    import pandas as pd
    true_labels = pd.read_table('../../split_seq_analysis/spinal_cluster_assignment.txt', header=None)
    true_labels = true_labels.as_matrix().astype(str)
    true_labels = true_labels.flatten()
    t0 = time.time()
    selected_k, labels = bnpy_select_clusters(data_subset)
    # there are 45 labels in the "true" dataset.
    print(selected_k)
    print('nmi: ' + str(nmi(true_labels, labels)))
    print('time: ' + str(time.time() - t0))

    # 8k single-cell
    data_5 = scipy.io.mmread('../../uncurl_test_datasets/10x_pure_pooled/data_8000_cells.mtx.gz')
    data_5 = scipy.sparse.csc_matrix(data_5)
    gene_subset = uncurl.max_variance_genes(data_5)
    data_subset = data_5[gene_subset, :]
    true_labels = np.loadtxt('../../uncurl_test_datasets/10x_pure_pooled/labels_8000_cells.txt').astype(int).flatten()
    t0 = time.time()
    selected_k, labels=bnpy_select_clusters(data_subset)
    print(selected_k)
    print('nmi: ' + str(nmi(true_labels, labels)))
    print('time: ' + str(time.time() - t0))
