import json
from pathlib import Path
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from nerMin import nerMin

from ClusterDefault import cluster as cluster_default
from ClusterAffpropHyperparams import cluster as cluster_affprop_hyperparams
from ClusterAffpropHyperparams2 import cluster as cluster_affprop_hyperparams2
from ClusterAffpropDistA1 import cluster as cluster_affprop_dist_a1
from ClusterAffpropKeepVectors import cluster as cluster_affprop_keep_vectors

from ClusterKMeans import cluster as cluster_kmeans
from ClusterDBScan import cluster as cluster_dbscan
from ClusterHDBScan import cluster as cluster_hdbscan
from ClusterOptics import cluster as cluster_optics

cluster_methods = {
    "Default": cluster_default,
    "AffpropHyperparams": cluster_affprop_hyperparams,
    "AffpropHyperparams2": cluster_affprop_hyperparams2,
    "AffpropDistA1": cluster_affprop_dist_a1,
    "AffpropKeepVectors": cluster_affprop_keep_vectors,
    "KMeans": cluster_kmeans,
    "DBScan": cluster_dbscan,
    "HDBScan": cluster_hdbscan,
    "Optics": cluster_optics,
    }

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


def recursive_find_docs(path: Path) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from recursive_find_docs(p)
        elif p.suffix == ".txt":
            yield p


def do_1_text(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    entities_sm = nerMin(text, "sm")
    entities_lg = nerMin(text, "lg")
    entities = list(entities_sm | entities_lg)

    for name, cluster_method in cluster_methods.items():
        points, clusters = cluster_method(entities)
        with open(path.parent / f"{path.stem}_{name}_points.json", "w", encoding="utf-8") as f:
            json.dump(points, f, ensure_ascii=False, indent=4, default=numpyToPythonType)
        with open(path.parent / f"{path.stem}_{name}_clusters.json", "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, indent=4, default=numpyToPythonType)


def main(path: Path) -> None:
    if path.is_dir():
        files = list(recursive_find_docs(path))
        with ThreadPoolExecutor(4) as executor:
            list(tqdm(executor.map(do_1_text, files), total=len(files)))

    elif path.is_file():
        do_1_text(path)

    else:
        raise ValueError(f"{path} is not a file or a directory")


if __name__ == "__main__":
    main(Path("corpus"))