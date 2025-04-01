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

def visualise_similarity_matrix(matrix, idxs):
    matrix = matrix[idxs][:, idxs]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)
    return fig

def make_relation_matrix(list, names):
    matrix = np.zeros((len(names), len(names)))
    for (e1, e2) in list:
        matrix[names.index(str(e1)), names.index(str(e2))] = 1
    return matrix

def make_located_in_same(located_in, regions):
    located_in_dict = {}
    located_in_same = []
    for (e1, e2) in located_in:
        if e2 not in regions:
            continue
        if e2 not in located_in_dict.keys():
            located_in_dict[e2] = [e1]
        else:
            located_in_dict[e2].append(e1)

    for hl in located_in_dict.keys():
        for e1 in located_in_dict[hl]:
            for e2 in located_in_dict[hl]:
                located_in_same.append((e1, e2))
                located_in_same.append((e2, e1))
    return located_in_same

SUBCONTINENTS = ['southern_asia', 'south-eastern_asia', 'eastern_asia', 'central_asia', 'western_asia', 'northern_africa', 'middle_africa', 'western_africa', 'easter_africa', 'southern_africa', 'northern_europe', 'western_europe', 'central_europe', 'eastern_europe', 'southern_europe', 'caribbean', 'northern_americas', 'central_america', 'south_america', 'polynesia', 'australia_and_new_zealand', 'melanesia', 'micronesia']
CONTINENTS = ['africa', 'americas', 'asia', 'europe', 'oceania']

def visualise_relations(clauses, names):
    names = [str(n) for n in names if (str(n) not in CONTINENTS + SUBCONTINENTS + ['locatedIn', 'neighborOf'])]

    # :-(countries(~(r),~(e1),~(e2)))
    relation_dict = {}
    for clause in clauses:
        head, body = clause.arguments
        if len(list(body.arguments)) > 0:
            continue
        r, e1, e2 = [a.arguments[0] for a in head.arguments]
        if str(r) not in relation_dict:
            relation_dict[str(r)] = [(e1,e2)]
        else:
            relation_dict[str(r)].append((e1,e2))
    figs = []

    located_in_same_continent = make_located_in_same(relation_dict['locatedIn'], CONTINENTS)
    located_in_same_subcontinent = make_located_in_same(relation_dict['locatedIn'], SUBCONTINENTS)
    matrix1 = make_relation_matrix(located_in_same_continent, list(names))
    matrix2 = make_relation_matrix(located_in_same_subcontinent, list(names))
    matrix1, matrix2, idx = argsort_sim_matrix(matrix1, matrix2, mode="max")

    located_in_same_continent_fig, ax = plt.subplots(figsize=(8, 6))
    _ = ax.matshow(matrix1, cmap='viridis')
    located_in_same_subontinent_fig, ax = plt.subplots(figsize=(8, 6))
    _ = ax.matshow(matrix1, cmap='viridis')
    # located_in_same_continent_fig.colorbar(cax)

    for r in relation_dict.keys():
        matrix = make_relation_matrix(relation_dict[r], list(names), r)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(matrix, cmap='viridis')
        fig.colorbar(cax)
        figs.append(fig)

    return figs+[located_in_same_continent_fig, located_in_same_subcontinent], list(relation_dict.keys())+['locatedInSameContinent','locatedInSameSubcontinent'], idx

def argsort_sim_matrix(sm, sm2, mode=max):
    sort_f = np.argmin if mode == "min" else np.argmax
    idx = [sort_f(np.sum(sm, axis=1) + np.sum(sm2, axis=1)/3)]  # a
    smis = []
    smis_2 = []
    for i in range(1, len(sm)):
        sm_i = sm[idx[-1]].copy()
        sm_i[idx] = np.inf
        smis += [sm_i]
        smis_2 += sm2[idx[-1]].copy()
        idx.append(np.sort_f(np.array(smis).mean(0) + np.array(smis_2).mean(0)/3))
    return sm[idx][:, idx],  sm2[idx][:, idx], idx