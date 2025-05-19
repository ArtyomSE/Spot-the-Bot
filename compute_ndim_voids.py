import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def find_node_cliques(adj_matrix, i, depth, cliques, curr_clique):
    for j in range(i + 1, adj_matrix.shape[0]):
        if adj_matrix[i, j] == 1:
            next_node_found = True
            for node in curr_clique[:-1]:
                if adj_matrix[j, node] == 0:
                    next_node_found = False

            if next_node_found == True:
                if depth == 1:
                    found_clique = tuple(curr_clique + [j])

                    clique_index = np.sum(
                        np.array(found_clique) * bases[-len(found_clique):]
                    ) + sizes[:len(found_clique)].sum()

                    global found_cliques
                    if found_cliques[clique_index] == 0:
                        cliques += [found_clique]
                        found_cliques[clique_index] = 1
                else:
                    cliques = find_node_cliques(
                        adj_matrix, j, depth - 1, cliques, curr_clique + [j]
                    )
    return cliques

def find_cliques(adj_matrix, ord):
    cliques = []
    for i in range(adj_matrix.shape[0]):
        node_cliques = find_node_cliques(adj_matrix, i, ord - 1, [], [i])
        if len(node_cliques) > 0:
            cliques += node_cliques
    return cliques


def compute_ndim_voids(data, epsilons, dim):
    n = data.shape[0]
    dist_matrix = pairwise_distances(data)
    
    bases = [n ** i for i in range(dim)]
    bases.reverse()
    sizes = [0] + [n ** i for i in range(1, dim)]
    found_cliques = np.zeros(n ** dim, dtype=np.int8)

    filtration = []
    for epsilon in epsilons:
        adj_matrix = np.zeros((n, n))
        adj_matrix[dist_matrix[:, :] <= epsilon] = 1
        adj_matrix -= np.eye(n)

        cliques = find_cliques(adj_matrix, dim)
        if len(cliques) > 0: filtration += cliques

    homologies = dict()
    for j in tqdm(np.arange(data.shape[0] - 1, -1, step=-1)):
        identity_elements_indices = np.where(boundary_matrix[:, 1] == j)[0]

        if identity_elements_indices.shape[0] > 1:
            leading_element = boundary_matrix[identity_elements_indices[0]]
            reducing_elements_indices = identity_elements_indices[1:]

            new_voids = set(reducing_elements_indices) - set(homologies.keys())
            for num in new_voids:
                homologies[num] = [np.copy(boundary_matrix[num])]

            for index in reducing_elements_indices:
                homologies[index].append(1 * np.copy(leading_element))

            boundary_matrix[reducing_elements_indices, 1] = leading_element[0]
            boundary_matrix[reducing_elements_indices] = np.sort(
                boundary_matrix[reducing_elements_indices]
            )

    return homologies