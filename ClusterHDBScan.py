from typing import List

from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from ClusterHelper import ClusterHelper
from errors import SmolBoi


def cluster(docs: List[str]) -> dict:
    """
    Cluster the documents
    :param docs: The documents to cluster
    :return: The clusters
    """
    V = CountVectorizer(ngram_range=(2, 4), analyzer='char', min_df=2)
    try:
        X = V.fit_transform(docs).toarray()
    except ValueError:
        raise SmolBoi("Not enough data to cluster")

    hdbscan = HDBSCAN(min_cluster_size=2, metric="cosine", store_centers="medoid")
    hdbscan.fit_predict(X)

    # points = []
    # for cluster_id in np.unique(hdbscan.labels_):
    #     centroid = np.mean(X_dist_svd[hdbscan.labels_ == cluster_id], axis=0)
    #     for i in np.where(hdbscan.labels_ == cluster_id)[0]:
    #         points.append(
    #             {
    #                 "text": docs[i],
    #                 "x_pos": X_dist_svd[i][0],
    #                 "y_pos": X_dist_svd[i][1],
    #                 "cluster": cluster_id,
    #                 "centroid": centroid,
    #                 "is_centroid": hdbscan.medoids_[cluster_id] == i
    #             }
    #         )
    #
    # clusters = {}
    # for cluster_id in np.unique(hdbscan.labels_):
    #     cluster = {docs[i] for i in range(len(docs)) if hdbscan.labels_[i] == cluster_id}
    #     clusters[str(cluster_id)] = cluster
    #
    # df_points = pd.DataFrame(points)
    # df_points_for_corr = df_points[cols_to_keep].copy()
    # df_points_for_corr['cluster_corrected'] = ""
    #
    # return points, clusters, df_points, df_points_for_corr
    return ClusterHelper.generate_whole_outputs(docs, X, hdbscan.labels_)


if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
