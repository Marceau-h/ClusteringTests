import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
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
    V = CountVectorizer(ngram_range=(2, 4), analyzer='char', min_df=2)
    X = V.fit_transform(docs).toarray()
    X_dist = cosine_distances(X)

    svd = SVD(n_components=2)
    X_dist_svd = svd.fit_transform(X_dist)

    hdbscan = HDBSCAN(min_cluster_size=2, metric="cosine", store_centers="medoid")
    hdbscan.fit_predict(X)

    points = []
    for cluster_id in np.unique(hdbscan.labels_):
        centroid = np.mean(X_dist_svd[hdbscan.labels_ == cluster_id], axis=0)
        for i in np.where(hdbscan.labels_ == cluster_id)[0]:
            points.append(
                {
                    "text": docs[i],
                    "x_pos": X_dist_svd[i][0],
                    "y_pos": X_dist_svd[i][1],
                    "cluster": cluster_id,
                    "centroid": centroid,
                    "is_centroid": hdbscan.medoids_[cluster_id] == i
                }
            )

    clusters = {}
    for cluster_id in np.unique(hdbscan.labels_):
        cluster = {docs[i] for i in range(len(docs)) if hdbscan.labels_[i] == cluster_id}
        clusters[str(cluster_id)] = cluster

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
