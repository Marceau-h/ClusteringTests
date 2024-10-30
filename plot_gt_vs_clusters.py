from pathlib import Path

import pandas as pd
from pydantic import model_validator
from tqdm.auto import tqdm
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator

raw_symbols = SymbolValidator().values

def from_float_to_pretty_string(x):
    return f"cluster_{int(x):03d}" if x != "" else "Pas de cluster"

def sort_the_pretty_strings(x):
    return int(x.split("_")[1]) if x != "Pas de cluster" else -100


def plot_clusters(
        df_gt,
        df_hyp,
        outp,
        livre="AIMARD_trappeurs",
        ocr="Kraken",
        gt="gt",
        model="affprop_keep_vectors2"
):
    df_all = pd.merge(
        df_gt,
        df_hyp,
        how='outer',
        on=['text'],
        suffixes=('_gt', '_hyp')
    ).fillna("")

    df_all = df_all[df_all.x_pos_hyp.notna()]
    df_all = df_all[df_all.y_pos_hyp.notna()]

    df_all["cluster_gt"] = df_all["cluster_gt"].apply(from_float_to_pretty_string)
    df_all["cluster_hyp"] = df_all["cluster_hyp"].apply(from_float_to_pretty_string)

    df_all = df_all.sort_values(by=["cluster_gt", "cluster_hyp"], key=lambda x: x.apply(sort_the_pretty_strings))

    # df_all.to_csv(f"outp/{livre}_{ocr}_{gt}_{model}.csv", index=False)

    # print(df_all.columns)
    # print(df_all.head())

    fig = px.scatter(
        df_all,
        x='x_pos_hyp',
        y='y_pos_hyp',
        color='cluster_gt',
        symbol='cluster_hyp',
        hover_data=['text'],

    )
    fig.update(layout_coloraxis_showscale=False, layout_showlegend=False)
    fig.update_traces(marker=dict(size=20))
    fig.update_layout(
        title=f"Comparaison des clusters formés par {gt} et {model} pour {livre} avec {ocr}",
        xaxis_title="",
        yaxis_title="",
    )
    # fig.update_layout(
    #     legend={
    #         "yanchor":"top",
    #         "y":0.99,
    #         "xanchor":"left",
    #         "x":0.01,
    #     }
    # )

    fig.show()
    fig.write_html(outp/f"{livre}_{ocr}_{gt}_{model}.html")

    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.png")
    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.jpg")
    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.jpeg")
    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.webp")
    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.svg")
    fig.write_image(outp/f"{livre}_{ocr}_{gt}_{model}.pdf")
    # fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.eps")

    fig.write_json(outp/f"{livre}_{ocr}_{gt}_{model}.json")

def main(
        path_to_gt: Path,
        path_to_hyp: Path,
        output_folder: Path,
        livre="AIMARD_trappeurs",
        ocr="Kraken",
        gt="gt",
        model="affprop_keep_vectors2"
):
    df_gt = pd.read_json(path_to_gt, lines=True)
    df_hyp = pd.read_json(path_to_hyp, lines=True)

    plot_clusters(df_gt, df_hyp, output_folder, livre, ocr, gt, model)


if __name__ == "__main__":
    outp = Path("outp_finale")
    outp.mkdir(exist_ok=True)

    gts = [
        "outp/AGUILAR_home-influence_Kraken_GT_df_points.jsonl",
        "outp/AGUILAR_home-influence_REF_GT_df_points.jsonl",
        "outp/AGUILAR_home-influence_Tesseract-PNG_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_PP_GT_df_points.jsonl",
        "outp/AIMARD_les-trappeurs_TesseractFra-PNG_GT_df_points.jsonl",
    ]

    gts = [Path(gt) for gt in gts]

    hyps = [
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/",
        "corpus_en/AGUILAR_home-influence/AGUILAR_REF/",
        "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Tesseract-PNG/",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_REF/",
        "corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_TesseractFra-PNG/",
    ]

    hyps = [Path(hyp) for hyp in hyps]

    for gt, hyp in tqdm(zip(gts, hyps)):
        livre = "_".join(gt.stem.split("_")[0:2])
        ocr = hyp.stem.split("_")[-1]
        for h in hyp.glob("*_df_points.jsonl"):
            model = h.stem.replace("_df_points.jsonl", "").split("_")[-1]
            main(
                gt,
                h,
                outp,
                livre=livre,
                ocr=ocr,
                gt="gt",
                model=model
            )



