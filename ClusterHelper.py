import json
from typing import List, Dict, Optional, Tuple, Any, Union, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD as SVD

class ClusterHelper:
    cols_to_keep = ['text', 'cluster']

    @staticmethod
    def numpyToPythonType(obj: Any) -> Union[int, float, list, bool, None]:
        """
        Convert numpy types to python types for json serialization
        :param obj: The object to convert
        :return: The converted object
        """
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

    @staticmethod
    def compare_vectors(v1: np.ndarray, v2: np.ndarray) -> int:
        """
        Compare two vectors and return the number of elements that are the same
        :param v1: The first vector
        :param v2: The second vector
        :return: The number of elements that are the same
        """
        v1 = v1[0]
        v2 = v2[0]
        return np.sum(np.logical_and(v1, v2))

    @staticmethod
    def points_from_cluster(
            docs: Iterable[str],
            X: np.ndarray,
            yhat: np.ndarray,
            excluded: Optional[Iterable[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Generate the points file from the cluster results
        :param docs: The documents (text)
        :param X: The matrix of the documents from the vectorizer (CountVectorizer for example)
        :param X_dist: The distance matrix of the documents (cosine_distances for example)
        :param yhat: The cluster labels
        :return:
        """
        if X.shape[1] != 2:
            svd = SVD(n_components=2)
            X_compressed = svd.fit_transform(X)
        else:
            X_compressed = X

        points = []
        for cluster_id in np.unique(yhat):
            centroid = np.mean(X[yhat == cluster_id], axis=0)
            for i in np.where(yhat == cluster_id)[0]:
                xpos, ypos = X_compressed[i]
                # assert xpos !=0, f"Xpos is 0 for {docs[i]}"
                # assert ypos !=0, f"Ypos is 0 for {docs[i]}"
                if xpos == 0 or ypos == 0:
                    print(f"Xpos or Ypos is 0 for {docs[i]}")

                points.append(
                    {
                        "text": docs[i],
                        "x_pos": xpos,
                        "y_pos": ypos,
                        "cluster": cluster_id,
                        "centroid": centroid,
                        "is_centroid": yhat[i] == i
                    }
                )

        if excluded is not None:
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
        return points

    @staticmethod
    def clusters_from_points(docs: List[str], yhat: np.ndarray, excluded: Optional[Iterable[str]]) -> Dict[str, set]:
        clusters = {}
        for cluster_id in np.unique(yhat):
            cluster = {docs[i] for i in range(len(docs)) if yhat[i] == cluster_id}
            clusters[str(cluster_id)] = cluster

        if excluded is not None:
            clusters["-1"] = excluded
        return clusters

    @staticmethod
    def generate_whole_outputs(
            docs: Iterable[str],
            X: np.ndarray,
            yhat: np.ndarray,
            *args,
            cols_to_keep: Optional[List[str]] = None,
            excluded: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Dict[str, any]], Dict[str, set], pd.DataFrame, pd.DataFrame]:
        if cols_to_keep is None:
            cols_to_keep = ClusterHelper.cols_to_keep

        points = ClusterHelper.points_from_cluster(docs, X, yhat, excluded)
        clusters = ClusterHelper.clusters_from_points(docs, yhat, excluded)

        df_points = pd.DataFrame(points)
        df_points_for_corr = df_points[cols_to_keep].copy()
        df_points_for_corr['cluster_corrected'] = ""

        return points, clusters, df_points, df_points_for_corr

    @staticmethod
    def main(cluster_method, filename: str) -> None:
        with open("sm_entities.json", "r", encoding="utf-8") as f:
            sm_entities = set(json.load(f))

        with open("lg_entities.json", "r", encoding="utf-8") as f:
            lg_entities = set(json.load(f))

        points, clusters, dp, dpc = cluster_method(list(sm_entities | lg_entities))

        with open(f"tests/points_{filename}.json", "w", encoding="utf-8") as f:
            json.dump(points, f, ensure_ascii=False, indent=4, default=ClusterHelper.numpyToPythonType)

        with open(f"tests/clusters_{filename}.json", "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, indent=4, default=ClusterHelper.numpyToPythonType)

        dp.to_json(f"tests/df_points_{filename}.jsonl", orient="records", lines=True)
        dpc.to_json(f"tests/df_points_for_corr_{filename}.jsonl", orient="records", lines=True)
        dpc.to_csv(f"tests/df_points_for_corr_{filename}.csv", index=False)

    @staticmethod
    def keep_vectors(matrix: np.ndarray) -> list:
        """
        Keep only the vectors that have at least one non-zero value, and return their indices
        :param matrix: The matrix to filter
        :return: the vectors indices to retain
        """
        return [i for i, v in enumerate(matrix) if np.any(v)]

