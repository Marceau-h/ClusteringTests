from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px

# gt = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_GT_df_points.jsonl"
#
# hyp1 = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropKeepVectors_df_points.jsonl"
# hyp2 = "corpus_en/AGUILAR_home-influence/AGUILAR_home-influence_OCR/AGUILAR_Kraken/AGUILAR_home-influence_Kraken_AffpropHyperparams2_df_points.jsonl"

gt = Path("outp/AIMARD_les-trappeurs_Kraken-base_GT_df_points.jsonl")
hyp1 = Path("corpus/AIMARD_TRAPPEURS/AIMARD-TRAPPEURS_OCR/AIMARD-TRAPPEURS_kraken/AIMARD_les-trappeurs_Kraken-base_AffpropHyperparams2_df_points.jsonl")

df_gt = pd.read_json(gt, lines=True)
df_hyp1 = pd.read_json(hyp1, lines=True)
# df_hyp2 = pd.read_json(hyp2, lines=True)

def plot_clusters(df_gt, df_hyp, names=('gt', 'hyp')):
    df_all = pd.merge(
        df_gt,
        df_hyp,
        how='outer',
        on=['text'],
        suffixes=('_gt', '_hyp')
    )

    # df_all = df_all[["x_pos_hyp" is not None, "y_pos_hyp" is not None]]
    df_all = df_all[df_all.x_pos_hyp.notna()]
    df_all = df_all[df_all.y_pos_hyp.notna()]

    print(df_all.columns)
    print(df_all.head())

    fig = px.scatter(
        df_all,
        x='x_pos_hyp',
        y='y_pos_hyp',
        color='cluster_gt',
        symbol='cluster_hyp',
        hover_data=['text']
    )
    fig.update_traces(marker=dict(size=20))
    fig.update_layout(
        title=f"Comparaison des clusters formés par {names[0]} et {names[1]}",
        xaxis_title="",
        yaxis_title="",
    )
    fig.show()
    fig.write_html(f"outp/{names[0]}_vs_{names[1]}.html")
    fig.write_image(f"outp/{names[0]}_vs_{names[1]}.png")
    fig.write_image(f"outp/{names[0]}_vs_{names[1]}.eps")

plot_clusters(df_gt, df_hyp1, names=('gt', 'affprop_keep_vectors2'))

# plot_clusters(df_gt, df_hyp2, names=('gt', 'affprop_hyperparams2'))
