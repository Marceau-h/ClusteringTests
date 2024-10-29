from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator

raw_symbols = SymbolValidator().values

# gt = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_GT_df_points.jsonl"
#
# hyp1 = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropKeepVectors_df_points.jsonl"
# hyp2 = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points.jsonl"

gt = Path("outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl")
hyp1 = Path("corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropHyperparams2_df_points.jsonl")

df_gt = pd.read_json(gt, lines=True)
df_hyp1 = pd.read_json(hyp1, lines=True)
# df_hyp2 = pd.read_json(hyp2, lines=True)

def from_float_to_pretty_string(x):
    return f"cluster_{int(x):03d}" if x != "" else "Pas de cluster"


def sort_the_pretty_strings(x):
    return int(x.split("_")[1]) if x != "Pas de cluster" else -100


def plot_clusters(
        df_gt,
        df_hyp,
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
    fig.write_html(f"outp/{livre}_{ocr}_{gt}_{model}.html")

    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.png")
    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.jpg")
    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.jpeg")
    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.webp")
    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.svg")
    fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.pdf")
    # fig.write_image(f"outp/{livre}_{ocr}_{gt}_{model}.eps")

    fig.write_json(f"outp/{livre}_{ocr}_{gt}_{model}.json")

plot_clusters(df_gt, df_hyp1, livre="AIMARD_trappeurs", ocr="Kraken", gt="gt", model="affprop_keep_vectors2")

# plot_clusters(df_gt, df_hyp2, names=('gt', 'affprop_hyperparams2'))
