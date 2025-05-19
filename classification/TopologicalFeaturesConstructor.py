import numpy as np
import pymorphy3
from sklearn.metrics import pairwise_distances
from collections import Counter

class TopologicalFeaturesConstructor:
    def __init__(self, dict_path, voids_embeddings_path, embeddings_len):
        self.embeddings = np.load(dict_path, allow_pickle=True).item()
        self.voids = np.load(voids_embeddings_path, allow_pickle=True).item()
        self.embeddings_len = embeddings_len

        self.lemmatizer = pymorphy3.MorphAnalyzer(lang='ru')
        self.forbidden_symbols = '0123456789,.[]{}()=≈>≥<≤+‡-_–±−*&№^%$#@¡!~;:ː§/\|¿?«»"\'•·≠…'
    
    def construct_topological_features(self, text):
        tokens = self.preprocess_text(text)
        semantic_space = self.get_semantic_space(tokens)
        topological_features = self.get_topological_features(semantic_space).reshape(1, -1)
        return topological_features
    
    def replace_symbols(self, text, symbols, replacement=''):
        for symbol in symbols:
            text = text.replace(symbol, replacement)
        return text

    def preprocess_text(self, text):
        text = text.lower()
        text = self.replace_symbols(text, self.forbidden_symbols)
        text = text.replace('\n', ' ')

        tokens = [self.lemmatizer.normal_forms(word)[0] for word in text.split()]
        return tokens
    
    def get_semantic_space(self, tokens):
        unique_tokens = set()
        semantic_space = []
        for i in range(len(tokens)):
            if tokens[i] in self.embeddings and tokens[i] not in unique_tokens:
                semantic_space.append(
                    self.embeddings[tokens[i]][-self.embeddings_len:]
                )
                unique_tokens.add(tokens[i])
        return np.array(semantic_space)

    def get_dist_to_centers(self, semantic_space, voids_centers):
        dist_to_centers = pairwise_distances(
            semantic_space, voids_centers[:, -self.embeddings_len:], metric='sqeuclidean'
        )
        dist_to_centers_mean = np.array(dist_to_centers.mean(axis=1)).reshape(-1, 1)
        return np.hstack((dist_to_centers, dist_to_centers_mean))

    def get_most_common_closest_void(self, min_dist):
        cnt = Counter(min_dist.argmin(axis=1))
        a = np.array([cnt[hn] for hn in range(min_dist.shape[1])]) / min_dist.shape[0]
        return np.hstack([a, [a.argmax()]])

    def get_dist(self, semantic_space, apply_func=np.min):
        dist_list = np.vstack([
            apply_func(pairwise_distances(
                semantic_space, void[:, -self.embeddings_len:], metric='sqeuclidean'
            ), axis=1) for void in self.voids.values()
        ])
        dist_list = np.vstack([dist_list, dist_list.mean(axis=0)])
        return dist_list.T

    def get_topological_features(self, semantic_space):
        voids_centers = np.vstack([void.mean(axis=0) for void in self.voids.values()])

        dist_to_centers_mean = np.mean(self.get_dist_to_centers(semantic_space, voids_centers), axis=0)
        min_dist = self.get_dist(semantic_space, np.min)
        max_dist_mean = np.mean(self.get_dist(semantic_space, np.max), axis=0)
        min_dist_to_closest_void = self.get_most_common_closest_void(min_dist[:, :-1])

        topological_features = np.hstack(
            [dist_to_centers_mean, np.mean(min_dist, axis=0), max_dist_mean, min_dist_to_closest_void]
        )

        return topological_features