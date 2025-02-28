from typing import List, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ClusterHelper import ClusterHelper
from errors import SmolBoi


def cluster(docs: List[str]) -> tuple[list[dict[str, Any]], dict[str, set], DataFrame, DataFrame]:
    """
    Cluster the documents
    :param docs: The documents to cluster
    :return: The clusters
    """
    treshold = 0.1

    V = CountVectorizer(ngram_range=(3, 4), analyzer='char', min_df=3)
    try:
        X = V.fit_transform(docs).toarray()
    except ValueError:
        raise SmolBoi("Not enough data to cluster")
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

    # points = []
    # for i, doc in enumerate(docs):
    #     points.append(
    #         {
    #             "text": doc,
    #             "x_pos": X_dist_svd[i][0],
    #             "y_pos": X_dist_svd[i][1],
    #             "cluster": best.labels_[i],
    #             "is_centroid": best.cluster_centers_[best.labels_[i]] == X_dist_svd[i]
    #         }
    #     )
    #
    # for word in wo_cluster:
    #     points.append(
    #         {
    #             "text": word,
    #             "x_pos": None,
    #             "y_pos": None,
    #             "cluster": -1,
    #             "is_centroid": False
    #         }
    #     )
    #
    # clusters = {}
    # for cluster_id in np.unique(best.labels_):
    #     cluster = {docs[i] for i in range(len(docs)) if best.labels_[i] == cluster_id}
    #     clusters[str(cluster_id)] = cluster

    points, clusters, df_points, df_points_for_corr = ClusterHelper.generate_whole_outputs(docs, X, best.labels_)

    clusters["-1"] = wo_cluster

    df_points = pd.DataFrame(points)
    df_points_for_corr = df_points[ClusterHelper.cols_to_keep].copy()
    df_points_for_corr['cluster_corrected'] = ""

    return points, clusters, df_points, df_points_for_corr


if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
