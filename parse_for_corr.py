from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def from_for_corr_and_points_to_gt_points(
        path_to_corr: Path,
        path_to_points: Path,
        path_to_save: Path
) -> None:
    print(f"Processing {path_to_corr.name} and {path_to_points.name} to {path_to_save.name}")

    df_corr = pd.read_csv(path_to_corr)
    assert df_corr.cluster_corrected.isna().sum() < df_corr.shape[0]
    df_corr.cluster_corrected.fillna(-1, inplace=True)

    df_points = pd.read_json(path_to_points, lines=True)

    df_joined = df_points.merge(df_corr, on='text', how='outer')
    del df_corr, df_points

    assert df_joined.cluster_x.equals(df_joined.cluster_y)

    df_joined.drop(columns=['cluster_y', "cluster_x"], inplace=True)
    df_joined.rename(columns={'cluster_corrected': 'cluster'}, inplace=True)

    df_joined.to_json(path_to_save, orient='records', lines=True)

if __name__ == "__main__":
    corrs = [
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points_for_corr.csv",
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points_for_corr.csv",
        "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points_for_corr.csv",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropKeepVectors_df_points_for_corr.csv",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG/AIMARD_les-trappeurs_TesseractFra-PNG_AffpropKeepVectors_df_points_for_corr.csv",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropKeepVectors_df_points_for_corr.csv",
    ]

    points = [
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points.jsonl",
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points.jsonl",
        "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points.jsonl",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropKeepVectors_df_points.jsonl",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG/AIMARD_les-trappeurs_TesseractFra-PNG_AffpropKeepVectors_df_points.jsonl",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropKeepVectors_df_points.jsonl",
    ]

    outputs = [
        "outp/AGUILAR_home-influence_Kraken_GT_df_points.jsonl",
        "outp/AGUILAR_home-influence_Tesseract-PNG_GT_df_points.jsonl",
        "outp/AGUILAR_home-influence_REF_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_TesseractFra-PNG_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_PP_GT_df_points.jsonl",
    ]

    for corr, point, output in tqdm(zip(corrs, points, outputs)):
        from_for_corr_and_points_to_gt_points(Path(corr), Path(point), Path(output))




