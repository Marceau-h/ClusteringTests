import json
import warnings
from pathlib import Path
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm
import pandas as pd

from nerMin import nerMin
from errors import SmolBoi, BigBoi

from ClusterDefault import cluster as cluster_default
from ClusterAffpropHyperparams import cluster as cluster_affprop_hyperparams
from ClusterAffpropHyperparams2 import cluster as cluster_affprop_hyperparams2
from ClusterAffpropDistA1 import cluster as cluster_affprop_dist_a1
from ClusterAffpropKeepVectors import cluster as cluster_affprop_keep_vectors

from ClusterKMeans import cluster as cluster_kmeans
from ClusterDBScan import cluster as cluster_dbscan
from ClusterHDBScan import cluster as cluster_hdbscan
from ClusterOptics import cluster as cluster_optics

from ClusterHelper import ClusterHelper

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



def recursive_find_docs(path: Path) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from recursive_find_docs(p)
        elif p.suffix == ".txt":
            yield p


def do_1_text(path: Path, lang:str='fr', resume:bool=True) -> None:
    error_file = path.parent / f"{path.stem}_error.txt"
    if resume:
        if error_file.exists():
            print(f"Skipping {path} because of previous error")
            return

        result_files = list(path.parent.glob(f"{path.stem}_*.json"))
        # if len(result_files) == len(cluster_methods) * 2:
        #     print(f"Skipping {path} because all results are already present")
        #     return

    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        entities_sm = nerMin(text=text, model="sm", lang=lang, enforce_nlp_length=True)
        entities_lg = nerMin(text=text, model="lg", lang=lang, enforce_nlp_length=True)
    except (BigBoi, SmolBoi) as e:
        print(f"Error for {path}: {e}")
        error_file.touch()
        return

    entities = list(entities_sm | entities_lg)

    for name, cluster_method in cluster_methods.items():
        points, clusters, dp, dpc = cluster_method(entities)
        with open(path.parent / f"{path.stem}_{name}_points.json", "w", encoding="utf-8") as f:
            json.dump(points, f, ensure_ascii=False, default=ClusterHelper.numpyToPythonType)
        with open(path.parent / f"{path.stem}_{name}_clusters.json", "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, default=ClusterHelper.numpyToPythonType)

        dp.to_json(path.parent / f"{path.stem}_{name}_df_points.jsonl", orient="records", lines=True)
        dpc.to_json(path.parent / f"{path.stem}_{name}_df_points_for_corr.jsonl", orient="records", lines=True)
        dpc.to_csv(path.parent / f"{path.stem}_{name}_df_points_for_corr.csv", index=False)

        dpc_to_friendly_csv(dpc, path.parent / f"{path.stem}_{name}_df_points_for_corr_friendly.csv")

def do_1_text_constructor(resume:bool=True):
    def do_1_text(path: Path, lang:str='fr') -> None:
        error_file = path.parent / f"{path.stem}_error.txt"
        if resume:
            if error_file.exists():
                print(f"Skipping {path} because of previous error")
                return

            result_files = list(path.parent.glob(f"{path.stem}_*.json"))
            if len(result_files) == len(cluster_methods) * 2:
                print(f"Skipping {path} because all results are already present")
                return

        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        try:
            entities_sm = nerMin(text=text, model="sm", lang=lang, enforce_nlp_length=True)
            entities_lg = nerMin(text=text, model="lg", lang=lang, enforce_nlp_length=True)
        except (BigBoi, SmolBoi) as e:
            print(f"Error for {path}: {e}")
            error_file.touch()
            return

        entities = list(entities_sm | entities_lg)

        for name, cluster_method in cluster_methods.items():
            points, clusters, dp, dpc = cluster_method(entities)
            with open(path.parent / f"{path.stem}_{name}_points.json", "w", encoding="utf-8") as f:
                json.dump(points, f, ensure_ascii=False, default=ClusterHelper.numpyToPythonType)
            with open(path.parent / f"{path.stem}_{name}_clusters.json", "w", encoding="utf-8") as f:
                json.dump(clusters, f, ensure_ascii=False, default=ClusterHelper.numpyToPythonType)

            dp.to_json(path.parent / f"{path.stem}_{name}_df_points.jsonl", orient="records", lines=True)
            dpc.to_json(path.parent / f"{path.stem}_{name}_df_points_for_corr.jsonl", orient="records", lines=True)
            dpc.to_csv(path.parent / f"{path.stem}_{name}_df_points_for_corr.csv", index=False)

            dpc_to_friendly_csv(dpc, path.parent / f"{path.stem}_{name}_df_points_for_corr_friendly.csv")
    return do_1_text

def dpc_to_friendly_csv(dpc: pd.DataFrame, path: Path) -> None:
    dpc = dpc.groupby("cluster")

    max_cluster = max(dpc.groups.keys())
    max_len = max(len(group) for group in dpc.groups.values())

    # with path.open("w", encoding="utf-8") as f:
    #     f.write("cluster,")
    #     for i in range(max_len):
    #         f.write(f"elem_{i},")
    #     f.write("\n")
    #
    #     for cluster, group in dpc:
    #         f.write(f"{cluster},")
    #         for i in range(max_len):
    #             try:
    #                 f.write(f"{group.iloc[i]['text']},")
    #             except IndexError:
    #                 f.write(",")
    #         f.write("\n")

    friendly_df = []
    for cluster, group in dpc:
        friendly_df.append(
            {
                "cluster": cluster,
                **{f"elem_{i}": group.iloc[i]["text"] if i < len(group) else None for i in range(max_len)}
            }
        )

    friendly_df = pd.DataFrame(friendly_df)
    friendly_df.to_csv(path, index=False)


def main(path: Path, lang:str='fr', resume:bool=True) -> None:
    warnings.simplefilter("ignore") # Ignore warnings in this scope
    print("Caution: The warnings will be ignored (too many warnings otherwise)")
    if path.is_dir():
        files = list(recursive_find_docs(path))
        # do_1_text = do_1_text_constructor(resume)
        # with ThreadPoolExecutor(4) as executor:
        #     list(tqdm(executor.map(do_1_text, files), total=len(files)))
        pbar = tqdm(files)
        for file in pbar:
            pbar.set_postfix(file=file)
            do_1_text(file, lang, resume)

    elif path.is_file():
        do_1_text(path, lang, resume)

    else:
        raise ValueError(f"{path} is not a file or a directory")

def reset_errors(path: Path) -> None:
    if path.is_dir():
        files = list(recursive_find_docs(path))
        for file in files:
            error_file = file.parent / f"{file.stem}_error.txt"
            if error_file.exists():
                error_file.unlink()

    elif path.is_file():
        error_file = path.parent / f"{path.stem}_error.txt"
        if error_file.exists():
            error_file.unlink()

    else:
        raise ValueError(f"{path} is not a file or a directory")

if __name__ == "__main__":
    corpus, lang = Path("corpus"), "fr"
    # corpus, lang = Path("corpus_en"), "en"
    # corpus, lang = Path("corpus_pt"), "pt"
    # reset_errors(corpus)
    main(corpus, lang, True)
