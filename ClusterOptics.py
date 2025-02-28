import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ClusterHelper import ClusterHelper
from errors import SmolBoi


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

    optics = OPTICS(min_samples=2, metric="precomputed")
    optics.fit_predict(X_dist)

    # points = []
    # for cluster_id in np.unique(optics.labels_):
    #     centroid = np.mean(X_dist_svd[optics.labels_ == cluster_id], axis=0)
    #     for i in np.where(optics.labels_ == cluster_id)[0]:
    #         points.append(
    #             {
    #                 "text": docs[i],
    #                 "x_pos": X_dist_svd[i][0],
    #                 "y_pos": X_dist_svd[i][1],
    #                 "cluster": cluster_id,
    #                 "centroid": centroid,
    #                 "is_centroid": False
    #             }
    #         )
    #
    # clusters = {}
    # for cluster_id in np.unique(optics.labels_):
    #     cluster = {docs[i] for i in range(len(docs)) if optics.labels_[i] == cluster_id}
    #     clusters[str(cluster_id)] = cluster
    #
    # df_points = pd.DataFrame(points)
    # df_points_for_corr = df_points[cols_to_keep].copy()
    # df_points_for_corr['cluster_corrected'] = ""
    #
    # return points, clusters, df_points, df_points_for_corr

    return ClusterHelper.generate_whole_outputs(docs, X, optics.labels_)


if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
