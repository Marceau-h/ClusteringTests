from typing import List

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
    V = CountVectorizer(ngram_range=(2, 2), analyzer='char')
    try:
        X = V.fit_transform(docs).toarray()
    except ValueError:
        raise SmolBoi("Not enough data to cluster")
    X_dist = cosine_distances(X)

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    affprop.fit_predict(-1 * X_dist)

    return ClusterHelper.generate_whole_outputs(docs, X, affprop.labels_)


if __name__ == "__main__":
    # Name of the script \wo the `Cluster` prefix and the .py extension
    filename = __file__.rsplit("/", 1)[1][7:-3]

    ClusterHelper.main(cluster, filename)
