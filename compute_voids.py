import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def compute_voids(data, epsilons, space_type):
    n = data.shape[0]
    dist_matrix = pairwise_distances(data)
    found_cliques = np.zeros((n, n), dtype=np.int8)

    filtration = []
    for epsilon in epsilons:
        adj_matrix = np.zeros((n, n), dtype=np.int8)
        adj_matrix[dist_matrix[:, :] <= radius] = 1
        adj_matrix -= np.eye(n, dtype=np.int8)

        edges = np.array(np.where(adj_matrix == 1)).T
        filtration += [
            tuple(edge) for edge in tqdm(
                edges[edges[:, 0] < edges[:, 1]]
            ) if found_cliques[edge[0], edge[1]] == 0
        ]
        found_cliques[adj_matrix[:, :] == 1] = 1

    filtration = sorted(filtration, key=lambda x: dist_matrix[x[0], x[1]])
    
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

    np.save(f'data/voids/voids_{space_type}.npy', homologies)
    return homologies