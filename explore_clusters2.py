from pathlib import Path

import polars as pl
import plotly.express as px

def make_one_fig(metric:str, lang:str, df:pl.DataFrame) -> px.box:
    if all(df["value"].is_null()):
        print(f"No values for {metric} by hypothesis for {lang}")
        return None

    title = f"{metric} by hypothesis for {lang}"
    max_value = df["value"].max()
    min_value = df["value"].min()

    y_axis_range = [0,1]
    if min_value < 0:
        y_axis_range[0] = min_value
    if max_value > 1:
        y_axis_range[1] = max_value

    fig = px.scatter(
        df.to_pandas(),
        x="ocr",
        y="value",
        color="hyp",
        labels={"value": metric[0], "hyp": "Hypothesis", "ocr": "Source"},
        symbol="hyp"
    )

    fig.update_layout(yaxis_range=y_axis_range, title=title, scattermode="group", scattergap=0.75)

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )

    fig.show()

    return fig



def main(results_df: pl.DataFrame, res_folder: str|Path="figs"):
    if isinstance(res_folder, str):
        res_folder = Path(res_folder)
    elif not isinstance(res_folder, Path):
        raise TypeError("res_folder must be a str or a Path")

    res_folder.mkdir(exist_ok=True)

    df_per_lang = results_df.group_by("lang")

    for lang, df_lang in df_per_lang:
        lang = lang[0]

        df_per_metric = df_lang.group_by("metric")

        for metric, df_metric in df_per_metric:
            metric = metric[0]

            df_metric = df_metric.filter(pl.col("hyp").ne("AffpropDistA1")) # Trash cluster

            fig = make_one_fig(metric, lang, df_metric)
            if fig is None:
                continue

            file = res_folder/f"{lang}_{metric}_scatter.html"
            fig.write_html(file)
            fig.write_image(file.with_suffix(".png"))
            fig.write_image(file.with_suffix(".webp"))


if __name__ == '__main__':
    res_folder = Path("figs")
    res_folder.mkdir(exist_ok=True)

    df = pl.read_parquet("evaluation_results/df.parquet")


    main(df, res_folder)

