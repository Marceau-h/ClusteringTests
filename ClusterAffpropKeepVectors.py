from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ClusterHelper import ClusterHelper
from errors import SmolBoi


def cluster(docs: List[str]):
    """
    Cluster the documents
    :param docs: The documents to cluster
    :return: The clusters
    """
    V = CountVectorizer(ngram_range=(3, 4), analyzer='char', min_df=2)
    try:
        X = V.fit_transform(docs).toarray()
    except ValueError:
        raise SmolBoi("Not enough data to cluster")

    ids = ClusterHelper.keep_vectors(X)
    X = X[ids]
    words = [docs[i] for i in ids]
    excluded = set(docs) - set(words)

    X_dist = cosine_distances(X)
    X_sim = 1 - X_dist

    affprop = AffinityPropagation(affinity="precomputed", damping=.5, random_state=None, max_iter=1000)
    affprop.fit_predict(X_sim)

    # points = []
    # for cluster_id in np.unique(affprop.labels_):
    #     centroid = np.mean(X_dist_svd[affprop.labels_ == cluster_id], axis=0)
    #     for i in np.where(affprop.labels_ == cluster_id)[0]:
    #         points.append(
    #             {
    #                 "text": words[i],
    #                 "x_pos": X_dist_svd[i][0],
    #                 "y_pos": X_dist_svd[i][1],
    #                 "cluster": cluster_id,
    #                 "centroid": centroid,
    #                 "is_centroid": affprop.cluster_centers_indices_[cluster_id] == i
    #             }
    #         )
    #
    # for word in excluded:
    #     points.append(
    #         {
    #             "text": word,
    #             "x_pos": None,
    #             "y_pos": None,
    #             "cluster": None,
    #             "centroid": None,
    #             "is_centroid": False
    #         }
    #     )
    #
    # clusters = {}
    # for cluster_id in np.unique(affprop.labels_):
    #     cluster = {words[i] for i in range(len(words)) if affprop.labels_[i] == cluster_id}
    #     clusters[str(cluster_id)] = cluster
    #
    # clusters["-1"] = excluded
    #
    # df_points = pd.DataFrame(points)
    # df_points_for_corr = df_points[ClusterHelper.cols_to_keep].copy()
    # df_points_for_corr['cluster_corrected'] = ""
    #
    # return points, clusters, df_points, df_points_for_corr

    return ClusterHelper.generate_whole_outputs(words, X, affprop.labels_, excluded=excluded)


if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
