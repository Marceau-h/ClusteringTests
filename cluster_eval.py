import json
from pathlib import Path

import polars as pl
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from tqdm.auto import tqdm

y_col = "cluster"
x_cols = ["x_pos", "y_pos"]

super_metrics = {
    "rand_score": rand_score,
    "adjusted_rand_score": adjusted_rand_score,
    "mutual_info_score": mutual_info_score,
    "normalized_mutual_info_score": normalized_mutual_info_score,
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "homogeneity_score": homogeneity_score,
    "completeness_score": completeness_score,
    "v_measure_score": v_measure_score,
    "fowlkes_mallows_score": fowlkes_mallows_score,
}

pos_metrics = {
    "silhouette_score": silhouette_score,
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score,
}


def do_metric(metric, yhat, y_or_X):
    try:
        if metric in super_metrics:
            return super_metrics[metric](y_or_X, yhat)
        elif metric in pos_metrics:
            return pos_metrics[metric](y_or_X, yhat)
        else:
            raise ValueError(f"Invalid metric: {metric}")
    except ValueError as ve:
        print(f"Error: {ve}")
        return None


def find_hyps(book_dir: Path):
    for path in book_dir.iterdir():
        if path.name.endswith("df_points.jsonl"):
            yield path


def do_one_ref(ref, hyp_dir):
    ref_df = pl.read_ndjson(ref)
    y = ref_df[y_col].to_numpy()

    res = {}
    pbar = tqdm(list(find_hyps(hyp_dir)))
    for hyp in pbar:
        if "GT" in hyp.name:
            pbar.write(f"Skipping {hyp.name} (GT)")
            continue

        pbar.set_description(hyp.name)
        hyp_df = pl.read_ndjson(hyp)

        yhat = hyp_df[y_col].to_numpy()
        coords = hyp_df[x_cols].to_numpy()

        try:
            dists = cosine_distances(coords)
        except ValueError:
            dists = None

        res[
            hyp.name.replace("_df_points.jsonl", "").split("_")[-1]
        ] = {
            **{
                metric: do_metric(metric, yhat, y)
                for metric in super_metrics
            },
            **(
                {
                    metric: do_metric(metric, yhat, dists)
                    for metric in pos_metrics
                } if dists is not None else {}
            )
        }

    return res


if __name__ == '__main__':
    # ref = Path("outp/AGUILAR_REF_GT_df_points.jsonl")
    # hyp_dir = Path("corpus_en/AGUILAR_home-influence/AGUILAR_REF")
    #
    # res = do_one_ref(ref, hyp_dir)
    #
    # with open("res.json", "w", encoding="utf-8") as f:
    #     json.dump(res, f)

    refs = (
        "AGUILAR_REF_GT_df_points.jsonl",
        "AGUILAR_Kraken_GT_df_points.jsonl",
        "AGUILAR_Tesseract-PNG_GT_df_points.jsonl",

        "AIMARD-TRAPPEURS_REF_GT_df_points.jsonl",
        "AIMARD-TRAPPEURS_kraken_GT_df_points.jsonl",
        "AIMARD-TRAPPEURS_TesseractFra-PNG_GT_df_points.jsonl",
    )

    refs = tuple(Path("outp") / ref for ref in refs)

    hyp_dirs = (
        "corpus_en/AGUILAR_home-influence/AGUILAR_REF",
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken",
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG",

        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG",
    )

    hyp_dirs = tuple(Path(hyp_dir) for hyp_dir in hyp_dirs)

    res = {}
    for ref, hyp_dir in zip(refs, hyp_dirs):
        res[ref.name] = do_one_ref(ref, hyp_dir)

    with open("res.json", "w", encoding="utf-8") as f:
        json.dump(res, f)



