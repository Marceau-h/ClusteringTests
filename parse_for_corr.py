from pathlib import Path
from typing import Iterable, List, Generator

import pandas as pd
from tqdm.auto import tqdm


def from_for_corr_and_points_to_gt_points(
        path_to_corr: Path,
        path_to_points: Path,
        path_to_save: Path,
        **bonus_fields
) -> None:
    print(f"Processing {path_to_corr.name} and {path_to_points.name} to {path_to_save.name}")

    try:
        df_corr = pd.read_csv(path_to_corr)
        assert df_corr.shape[1] == 3
    except (pd.errors.ParserError, AssertionError):
        df_corr = pd.read_csv(path_to_corr, sep=';')

    assert df_corr.cluster_corrected.isna().sum() < df_corr.shape[0]
    df_corr.cluster_corrected.fillna(-1, inplace=True)

    df_points = pd.read_json(path_to_points, lines=True)

    df_joined = df_points.merge(df_corr, on='text', how='outer')
    # del df_corr, df_points

    # assert df_joined.cluster_x.equals(df_joined.cluster_y)

    df_joined.drop(columns=['cluster_y', "cluster_x"], inplace=True)
    df_joined.rename(columns={'cluster_corrected': 'cluster'}, inplace=True)

    for k, v in bonus_fields.items():
        df_joined[k] = v

    df_joined.to_json(path_to_save, orient='records', lines=True)


def corr_finder(root: Path, key: str, suffix: str, exclude:str|Iterable[str]) -> Generator[Path, None, None]:
    if isinstance(exclude, str):
        exclude = {exclude}
    elif isinstance(exclude, Iterable):
        exclude = set(exclude)
    else:
        raise ValueError(f"exclude must be a str or an Iterable, got {exclude}")

    for path in root.iterdir():
        if path.is_dir():
            yield from corr_finder(path, key, suffix, exclude)
        elif key in path.name and path.suffix == suffix and not any(excl in path.name for excl in exclude):
            yield path

def keep_good_corrs(corrs: Iterable[Path]) -> List[Path]:
    per_parent = {}
    for corr in corrs:
        if corr.parent not in per_parent:
            per_parent[corr.parent] = []
        per_parent[corr.parent].append(corr)

    res = []
    for parent, corrs in per_parent.items():
        if len(corrs) > 1:
            print(f"Multiple corrs for {parent.name}: {corrs}")

            longest_name = max(corrs, key=lambda x: len(x.name))
            print(f"Keeping {longest_name}")
            res.append(longest_name)
        else:
            res.append(corrs[0])

    return res

def point_from_corr(corr: Path) -> Path:
    return corr.parent / f"{corr.with_suffix('.jsonl').name.split('df_points_for_corr')[0].strip('_')}_df_points.jsonl"

def all_points_from_corrs(corrs: List[Path]) -> List[Path]:
    return [point_from_corr(corr) for corr in corrs]

def outp_from_corr(outp_dir: Path, corr: Path) -> Path:
    return outp_dir / f"{corr.parent.name}_GT_df_points.jsonl"

def all_outps_from_corrs(outp_dir: Path, corrs: List[Path]) -> List[Path]:
    return [outp_from_corr(outp_dir, corr) for corr in corrs]


if __name__ == "__main__":
    # corrs = [
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points_for_corr.csv",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points_for_corr.csv",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points_for_corr.csv",
    #     "corpus (Copie)/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropKeepVectors_df_points_for_corr.csv",
    #     "corpus (Copie)/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG/AIMARD_les-trappeurs_TesseractFra-PNG_AffpropKeepVectors_df_points_for_corr.csv",
    #     "corpus (Copie)/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropKeepVectors_df_points_for_corr.csv",
    # ]
    #
    # points = [
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points.jsonl",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points.jsonl",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points.jsonl",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropHyperparams2_df_points.jsonl",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG/AIMARD_les-trappeurs_TesseractFra-PNG_AffpropHyperparams2_df_points.jsonl",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropHyperparams2_df_points.jsonl",
    # ]
    #
    # outputs = [
    #     "outp/AGUILAR_home-influence_Kraken_GT_df_points.jsonl",
    #     "outp/AGUILAR_home-influence_Tesseract-PNG_GT_df_points.jsonl",
    #     "outp/AGUILAR_home-influence_REF_GT_df_points.jsonl",
    #     "outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl",
    #     "outp/AIMARD_les-trappeurs_TesseractFra-PNG_GT_df_points.jsonl",
    #     "outp/AIMARD_les-trappeurs_PP_GT_df_points.jsonl",
    # ]

    corrs_en = corr_finder(Path("corpus_en"), "OK2", ".csv", "friendly")
    corrs_fr = corr_finder(Path("corpus"), "OK", ".csv", {"friendly", "note"})
    corrs_en = keep_good_corrs(corrs_en)
    corrs_fr = keep_good_corrs(corrs_fr)

    corrs = corrs_en + corrs_fr
    print(*corrs, sep="\n")

    points = all_points_from_corrs(corrs)

    outputs = all_outps_from_corrs(Path("outp"), corrs)

    bonus_fields = [
        {"lang": "en"}
    ] * len(corrs_en) + [
        {"lang": "fr"}
    ] * len(corrs_fr)


    # corrs = [
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points_for_corr-annot_OK2.csv",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropKeepVectors_df_points_for_corr-annot-OK-note.csv",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropKeepVectors_df_points_for_corr_friendly_corr-annot-OK.csv",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points_for_corr-annot_OK2.csv",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points_for_corr-annot-OK2.csv",
    # ]
    #
    # points = [
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/AGUILAR_home-influence_Tesseract-PNG_AffpropHyperparams2_df_points.jsonl",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropKeepVectors_df_points.jsonl",
    #     "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/AIMARD_les-trappeurs_PP_AffpropKeepVectors_df_points.jsonl",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points.jsonl",
    #     "corpus_en/AGUILAR_home-influence/AGUILAR_REF/AGUILAR_home-influence_REF_AffpropHyperparams2_df_points.jsonl",
    # ]
    #
    # outputs = [
    #     "outp/AGUILAR_home-influence_Tesseract-PNG_GT_df_points.jsonl",
    #     "outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl",
    #     "outp/AIMARD_les-trappeurs_PP_GT_df_points.jsonl",
    #     "outp/AGUILAR_home-influence_Kraken_GT_df_points.jsonl",
    #     "outp/AGUILAR_home-influence_REF_GT_df_points.jsonl",
    # ]

    for corr, point, output, bf in tqdm(zip(corrs, points, outputs, bonus_fields), total=len(corrs)):
        from_for_corr_and_points_to_gt_points(Path(corr), Path(point), Path(output), **bf)
