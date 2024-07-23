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


def cluster(docs: List[str]) -> dict:
    """
    Cluster the documents
    :param docs: The documents to cluster
    :return: The clusters
    """
    V = CountVectorizer(ngram_range=(3, 4), analyzer='char', min_df=3)
    X = V.fit_transform(docs).toarray()
    X_dist = cosine_distances(X)

    svd = SVD(n_components=2)
    X_dist_svd = svd.fit_transform(X_dist)

    affprop = AffinityPropagation(affinity="precomputed", damping=.5, random_state=None, max_iter=1000)
    affprop.fit_predict(-1 * X_dist)

    points = []
    for cluster_id in np.unique(affprop.labels_):
        centroid = np.mean(X_dist_svd[affprop.labels_ == cluster_id], axis=0)
        for i in np.where(affprop.labels_ == cluster_id)[0]:
            points.append(
                {
                    "text": docs[i],
                    "x_pos": X_dist_svd[i][0],
                    "y_pos": X_dist_svd[i][1],
                    "cluster": cluster_id,
                    "centroid": centroid,
                    "is_centroid": affprop.cluster_centers_indices_[cluster_id] == i
                }
            )

    clusters = {}
    for cluster_id in np.unique(affprop.labels_):
        cluster = {docs[i] for i in range(len(docs)) if affprop.labels_[i] == cluster_id}
        clusters[str(cluster_id)] = cluster

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
