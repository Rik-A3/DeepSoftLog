import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import viridis
from sklearn import manifold

def visualise_embeddings(embeddings, names):
    LLE =  manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
    proj_emb = LLE.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj_emb[:, 0], proj_emb[:, 1], cmap=viridis())
    plt.legend(handles=scatter.legend_elements()[0], labels=names)
    plt.title("LLE Projection of Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    return fig

def visualise_similarity_matrix(matrix):
    matrix = argsort_sim_matrix(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)
    return fig

def argsort_sim_matrix(sm):
    idx = [np.argmin(np.sum(sm, axis=1))]  # a
    for i in range(1, len(sm)):
        sm_i = sm[idx[-1]].copy()
        sm_i[idx] = np.inf
        idx.append(np.argmin(sm_i))  # b
    return sm[idx][:, idx]