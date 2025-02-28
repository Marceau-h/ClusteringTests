import json
from pathlib import Path

import polars as pl
import plotly.express as px

def make_df(res):
    df = []

    for book, book_res in res.items():
        for hyp, hyp_res in book_res.items():
            for metric, metric_res in hyp_res.items():
                df.append({
                    "book": book,
                    "hyp": hyp,
                    "metric": metric,
                    "value": metric_res
                })

    return pl.DataFrame(df)


def make_one_fig(metric, df):
    title = f"{metric} by hypothesis"
    max_value = df["value"].max()
    min_value = df["value"].min()

    y_axis_range = [0,1]
    if min_value < 0:
        y_axis_range[0] = min_value
    if max_value > 1:
        y_axis_range[1] = max_value


    fig = px.box(df, y="value", color="hyp", title=title, labels={"value": metric, "hyp": "Hypothesis"})
    fig.update_layout(yaxis_range=y_axis_range)

    fig.show()

    return fig

def make_figs(df, res_folder: str|Path="figs"):
    if isinstance(res_folder, str):
        res_folder = Path(res_folder)
    elif not isinstance(res_folder, Path):
        raise TypeError("res_folder must be a str or a Path")

    res_folder.mkdir(exist_ok=True)

    df = df.filter(pl.col("hyp").ne("AffpropDistA1")) # Trash cluster

    for metric, metric_df in df.group_by("metric"):
        fig = make_one_fig(metric[0], metric_df)

        file = res_folder / f"{metric[0]}.html"
        fig.write_html(file)
        fig.write_image(file.with_suffix(".png"))
        fig.write_image(file.with_suffix(".webp"))


if __name__ == '__main__':
    with open("res.json") as f:
        res = json.load(f)

    df = make_df(res)
    print(df.head(20))

    make_figs(df)


