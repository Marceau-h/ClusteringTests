from typing import List

from sklearn.cluster import DBSCAN
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
    V = CountVectorizer(ngram_range=(2, 4), analyzer='char', min_df=2)
    try:
        X = V.fit_transform(docs).toarray()
    except ValueError:
        raise SmolBoi("Not enough data to cluster")
    X_dist = cosine_distances(X)

    dbscan = DBSCAN(eps=0.1, min_samples=2, metric="precomputed")
    dbscan.fit_predict(X_dist)
    return ClusterHelper.generate_whole_outputs(docs, X, dbscan.labels_)



if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
