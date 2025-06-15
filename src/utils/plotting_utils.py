import random
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import cycle
from plotly.colors import n_colors
import pandas as pd
from statsmodels.tsa.stattools import pacf, acf
import plotly.graph_objects as go
import warnings
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")
from itertools import product

import matplotlib.pyplot as plt


def make_lines_greyscale(fig):
    colors = cycle(
        list(
            set(
                n_colors(
                    "rgb(100, 100, 100)", "rgb(200, 200, 200)", 2 + 1, colortype="rgb"
                )
            )
        )
    )
    for d in fig.data:
        d.line.color = next(colors)
    return fig


def two_line_plot_secondary_axis(
    x,
    y1,
    y2,
    y1_name="y1",
    y2_name="y2",
    title="",
    legends=None,
    xlabel="Time",
    ylabel="Value",
    greyscale=False,
    dash_secondary=False,
):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_name),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            name=y2_name,
            line=dict(dash="dash") if dash_secondary else None,
        ),
        secondary_y=True,
    )
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
            autosize=False,
            width=900,
            height=500,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
            title_text=title,
        titlefont={"size": 20},
        legend_title=None,
            yaxis=dict(
                title_text=ylabel,
                titlefont=dict(size=12),
            ),
            xaxis=dict(
                title_text=xlabel,
                titlefont=dict(size=12),
        ),
    )
    if greyscale:
        fig = make_lines_greyscale(fig)
    return fig

def multiple_line_plot_secondary_axis(df, x, primary, secondary, color_or_linetype, title="", use_linetype=False, greyscale=False):
    df = pd.pivot_table(df, index=x, columns=color_or_linetype, values=[primary, secondary]).reset_index()
    df.columns = [str(c1)+"_"+str(c2) if c2!="" else c1 for c1, c2 in df.columns]
    primary_columns = sorted([c for c in df.columns if primary in c])
    secondary_columns = sorted([c for c in df.columns if secondary in c])
    if use_linetype:
        colors = ["solid","dash","dot","dashdot"]
    else:
        colors = px.colors.qualitative.Plotly
        if len(primary_columns)>=len(colors):
            colors = px.colors.qualitative.Light24
    assert len(primary_columns)<=len(colors)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    for c, color in zip(primary_columns, colors[:len(primary_columns)]):
        fig.add_trace(
            go.Scatter(x=df[x], y=df[c], name=c, line=dict(dash=color) if use_linetype else dict(color=color)),
            secondary_y=False,
        )
    for c, color in zip(secondary_columns, colors[:len(primary_columns)]):
        fig.add_trace(
            go.Scatter(x=df[x], y=df[c], name=c, line=dict(dash=color) if use_linetype else dict(color=color)),
            secondary_y=True,
        )
    # Add figure title
    fig.update_layout(
        title_text=title
    )
    if greyscale:
#         colors = cycle(list(set(px.colors.sequential.Greys[1:])))
        colors = cycle(list(set(n_colors('rgb(100, 100, 100)', 'rgb(200, 200, 200)', 2+1, colortype='rgb'))))
        for d in fig.data:
            d.line.color = next(colors)
    return fig

def hex_to_rgb(hex):
    if hex.startswith("#"):
        hex = hex.lstrip("#")
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def plot_autocorrelation(series,vertical=False, figsize=(500, 900), **kwargs):
    if "qstat" in kwargs.keys():
        warnings.warn("`qstat` for acf is ignored as it has no impact on the plots")
        kwargs.pop("qstat")
    acf_args = ["adjusted","nlags", "fft", "alpha", "missing"]
    pacf_args = ["nlags","method","alpha"]
    if "nlags" not in kwargs.keys():
        nobs = len(series)
        kwargs['nlags'] = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    kwargs['fft'] = True
    acf_kwargs = {k:v for k,v in kwargs.items() if k in acf_args}
    pacf_kwargs = {k:v for k,v in kwargs.items() if k in pacf_args}
    acf_array = acf(series, **acf_kwargs)
    pacf_array = pacf(series, **pacf_kwargs)
    is_interval = False
    if "alpha" in kwargs.keys():
        acf_array, _ = acf_array
        pacf_array, _ = pacf_array
    x_ = np.arange(1,len(acf_array))
    rows, columns = (2, 1) if vertical else (1,2)
    fig = make_subplots(
            rows=rows, cols=columns, shared_xaxes=True, shared_yaxes=False, subplot_titles=['Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)']
        )
    #ACF
    row, column = 1, 1
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,acf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(acf_array))]
    fig.append_trace(go.Scatter(x=x_, y=acf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    #PACF
    row, column = (2,1) if vertical else (1,2)
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,pacf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(pacf_array))]
    fig.append_trace(go.Scatter(x=x_, y=pacf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    fig.update_traces(showlegend=False)
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(
            autosize=False,
            width=figsize[1],
            height=figsize[0],
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            titlefont={
                "size": 20
            },
            legend_title = None,
            yaxis=dict(
                titlefont=dict(size=12),
            ),
            xaxis=dict(
                titlefont=dict(size=12),
            )
        )
    return fig

def show_plotly_swatches():
    fig = px.colors.qualitative.swatches()
    fig.show()

def plot_correlation_plot(df, title="Heatmap", num_decimals=2, figsize=(200,200)):
    df = df.round(num_decimals)
    mask = np.triu(np.ones_like(df, dtype=bool))
    df_mask = df.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                      x=df_mask.columns.tolist(),
                                      y=df_mask.columns.tolist(),
                                      colorscale=px.colors.diverging.RdBu,
                                      hoverinfo="none", #Shows hoverinfo for null values
                                      showscale=True, ygap=1, xgap=1
                                     )
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title_text=title, 
        title_x=0.5, 
        width=figsize[0], 
        height=figsize[1],
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    return fig


# For plotting nixtla output dataframes
def plot_grid(
    df_train,
    df_test=None,
    plot_random=False,
    model=None,
    level=None,
    last_n_train=100,
):
    fig, axes = plt.subplots(4, 2, figsize=(24, 14))

    unique_ids = df_train["unique_id"].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"

    if plot_random:
        unique_ids = random.sample(list(unique_ids), k=8)
    else:
        unique_uids = unique_ids[:8]

    for uid, (idx, idy) in zip(unique_ids, product(range(4), range(2))):
        train_uid = df_train.query("unique_id == @uid")
        if last_n_train is not None:
            train_uid = train_uid.iloc[-last_n_train:]
        axes[idx, idy].plot(
            train_uid["ds"],
            train_uid["y"],
            label="y_train",
            linestyle="-",
            color="tab:blue",
        )
        if df_test is not None:
            max_ds = train_uid["ds"].max()
            test_uid = df_test.query("unique_id == @uid")
            model_col = (
                f"{model}-median"
                if f"{model}-median" in test_uid.columns
                else f"{model}"
            )
            for col in ["y", model_col, "y_test"]:
                if col in test_uid:
                    if col in ["y", "y_test"]:
                        axes[idx, idy].plot(
                            test_uid["ds"],
                            test_uid[col],
                            label=col,
                            linestyle=(0, (1, 1)),
                            color="tab:purple",
                        )
                    elif col == model_col:
                        axes[idx, idy].plot(
                            test_uid["ds"],
                            test_uid[col],
                            label=col,
                            linestyle=(5, (10, 3)),
                            color="black",
                        )
                    else:
                        axes[idx, idy].plot(
                            test_uid["ds"],
                            test_uid[col],
                            label=col,
                            linestyle=":",
                            color="tab:cyan",
                        )
            if level is not None:
                for l, alpha in zip(sorted(level), [0.5, 0.4, 0.35, 0.2]):
                    axes[idx, idy].fill_between(
                        test_uid["ds"],
                        test_uid[f"{model}-lo-{l}"],
                        test_uid[f"{model}-hi-{l}"],
                        alpha=alpha,
                        color="tab:orange",
                        label=f"{model}_level_{l}",
                    )
        axes[idx, idy].set_title(f"M4 Hourly: {uid}")
        axes[idx, idy].set_xlabel("Timestamp [t]")
        axes[idx, idy].set_ylabel("Target")
        axes[idx, idy].legend(loc="upper left")
        axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(20))
        axes[idx, idy].grid()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
