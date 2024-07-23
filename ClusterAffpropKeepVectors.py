import json
from typing import List

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances


def numpyToPythonType(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void):
        return None
    elif isinstance(obj, set):
        return list(obj)

    raise TypeError


def keep_vectors(matrix: np.ndarray) -> list:
    """
    Keep only the vectors that have at least one non-zero value, and return their indices
    :param matrix: The matrix to filter
    :return: the vectors indices to retain
    """
    return [i for i, v in enumerate(matrix) if np.any(v)]



def cluster(docs: List[str]) -> dict:
    """
    Cluster the documents
    :param docs: The documents to cluster
    :return: The clusters
    """
    V = CountVectorizer(ngram_range=(3, 4), analyzer='char', min_df=2)
    X = V.fit_transform(docs).toarray()

    ids = keep_vectors(X)
    X = X[ids]
    words = [docs[i] for i in ids]
    excluded = set(docs) - set(words)

    X_dist = cosine_distances(X)
    X_sim = 1 - X_dist

    svd = SVD(n_components=2)
    X_dist_svd = svd.fit_transform(X_sim)

    affprop = AffinityPropagation(affinity="precomputed", damping=.5, random_state=None, max_iter=1000)
    affprop.fit_predict(X_sim)

    points = []
    for cluster_id in np.unique(affprop.labels_):
        centroid = np.mean(X_dist_svd[affprop.labels_ == cluster_id], axis=0)
        for i in np.where(affprop.labels_ == cluster_id)[0]:
            points.append(
                {
                    "text": words[i],
                    "x_pos": X_dist_svd[i][0],
                    "y_pos": X_dist_svd[i][1],
                    "cluster": cluster_id,
                    "centroid": centroid,
                    "is_centroid": affprop.cluster_centers_indices_[cluster_id] == i
                }
            )

    for word in excluded:
        points.append(
            {
                "text": word,
                "x_pos": None,
                "y_pos": None,
                "cluster": None,
                "centroid": None,
                "is_centroid": False
            }
        )


    clusters = {}
    for cluster_id in np.unique(affprop.labels_):
        cluster = {words[i] for i in range(len(words)) if affprop.labels_[i] == cluster_id}
        clusters[str(cluster_id)] = cluster

    clusters["-1"] = excluded

    return points, clusters


if __name__ == "__main__":
    with open("sm_entities.json", "r", encoding="utf-8") as f:
        sm_entities = set(json.load(f))

    with open("lg_entities.json", "r", encoding="utf-8") as f:
        lg_entities = set(json.load(f))

    points, clusters = cluster(list(sm_entities | lg_entities))

    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    with open(f"tests/points_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=4, default=numpyToPythonType)

    with open(f"tests/clusters_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=4, default=numpyToPythonType)
