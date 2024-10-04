import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

cols_to_keep = ['text', 'cluster']


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
    treshold = 0.1

    V = CountVectorizer(ngram_range=(3, 4), analyzer='char', min_df=3)
    X = V.fit_transform(docs).toarray()
    X_dist = cosine_distances(X)

    svd = SVD(n_components=2)
    X_dist_svd = svd.fit_transform(X_dist)

    max_clusters = len(docs) // 3

    best = None
    best_silhouette = -1
    for n in range(2, max_clusters):
        kmeans = MiniBatchKMeans(n_clusters=n, random_state=0)
        kmeans.fit_predict(X_dist_svd)
        silhouette = np.mean(kmeans.inertia_)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best = kmeans

    wo_cluster = []
    for cluster_id in np.unique(best.labels_):
        centroid = best.cluster_centers_[cluster_id]
        centroid_norm = centroid / np.linalg.norm(centroid)
        exemplar = docs[np.argmax(centroid)]
        # cluster = np.unique(words[np.nonzero(best.labels_ == cluster_id)])
        cluster = []
        for i in np.nonzero(best.labels_ == cluster_id)[0]:
            word = docs[i]
            if word == exemplar:
                continue
            word_vector = X_dist_svd[i]
            word_vector_norm = word_vector / np.linalg.norm(word_vector)
            similarity = np.dot(centroid_norm, word_vector_norm)
            if similarity > treshold:
                cluster.append(word)
            else:
                wo_cluster.append(word)

    points = []
    for i, doc in enumerate(docs):
        points.append(
            {
                "text": doc,
                "x_pos": X_dist_svd[i][0],
                "y_pos": X_dist_svd[i][1],
                "cluster": best.labels_[i],
                "is_centroid": best.cluster_centers_[best.labels_[i]] == X_dist_svd[i]
            }
        )

    for word in wo_cluster:
        points.append(
            {
                "text": word,
                "x_pos": None,
                "y_pos": None,
                "cluster": -1,
                "is_centroid": False
            }
        )

    clusters = {}
    for cluster_id in np.unique(best.labels_):
        cluster = {docs[i] for i in range(len(docs)) if best.labels_[i] == cluster_id}
        clusters[str(cluster_id)] = cluster

    clusters["-1"] = wo_cluster

    df_points = pd.DataFrame(points)
    df_points_for_corr = df_points[cols_to_keep].copy()
    df_points_for_corr['cluster_corrected'] = ""

    return points, clusters, df_points, df_points_for_corr


if __name__ == "__main__":
    with open("sm_entities.json", "r", encoding="utf-8") as f:
        sm_entities = set(json.load(f))

    with open("lg_entities.json", "r", encoding="utf-8") as f:
        lg_entities = set(json.load(f))

    points, clusters, dp, dpc = cluster(list(sm_entities | lg_entities))

    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    with open(f"tests/points_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=4, default=numpyToPythonType)

    with open(f"tests/clusters_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=4, default=numpyToPythonType)


    dp.to_json(f"tests/df_points_{filename}.jsonl", orient="records", lines=True)
    dpc.to_json(f"tests/df_points_for_corr_{filename}.jsonl", orient="records", lines=True)
    dpc.to_csv(f"tests/df_points_for_corr_{filename}.csv", index=False)
