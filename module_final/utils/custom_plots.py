from typing import Optional

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_stacked_timeseries(data: pd.DataFrame,
                            title: str = '',
                            suffixes: Optional[tuple] = None,
                            anomaly: Optional[pd.Series] = None,
                            detect: Optional[pd.Series] = None,
                            height: Optional[int] = None,
                            ) -> None:
    '''
    Plotting time series from dataframe in stacked subplots style with shared time axe.
    Time series can group by suffix.
    Anomaly and detect intervals can shown on subplots.
    '''

    def plot_vrect(figr: go.Figure,
                   series: pd.Series,
                   vrect_type: str,
                   ) -> None:
        # plot vrect area using series of states
        if vrect_type == 'anomaly':
            fillcolor = 'red'
            position = 'top'
            prefix = 'A'
        elif vrect_type == 'detect':
            fillcolor = 'green'
            position = 'bottom'
            prefix = 'D'
        else:
            raise ValueError(f'Wrong vrect type {vrect_type}')
        series_ = series.copy()
        index_delta = series.index[-1] - series.index[-2]
        series_[series.index[-1] + index_delta] = 0  # always finished by normal state
        state = 0
        vrect = {'x0': None, 'x1': None, 'annotation_text': None}
        for i in series_.index:
            if series_[i] != state:
                # value changed
                if state != 0:
                    # this anomaly finished
                    figr.add_vrect(**vrect,
                                   annotation_position='outside ' + position,
                                   fillcolor=fillcolor,
                                   opacity=0.25,
                                   )
                # next anomaly
                state = series_[i]
                vrect['x0'] = i
                vrect['annotation_text'] = prefix + str(series_[i])
            vrect['x1'] = i

    # make figure with subplots
    if suffixes is not None:
        n_subplots = len(suffixes) + max(0 if c.endswith(suffixes) else 1 for c in data.columns)
    else:
        suffixes = ()
        n_subplots = 1
    fig = make_subplots(rows=n_subplots,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_titles=suffixes + ('other',),
                        x_title=data.index.name,
                        y_title=title,
                        )
    # plot subplots using suffixes
    for i_subplot, suffix in enumerate(suffixes):
        sfx_columns = [c for c in data.columns if c.endswith(suffix)]
        for k in sfx_columns:
            fig.add_scatter(y=data[k],
                            name=k,
                            row=i_subplot + 1,
                            col=1,
                            )
    # plot other in last subplots
    sfx_columns = [c for c in data.columns if not c.endswith(suffixes)]
    for k in sfx_columns:
        fig.add_scatter(y=data[k],
                        name=k,
                        row=n_subplots,
                        col=1,
                        )
    # plot anomaly area if possible
    if anomaly is not None and max(anomaly):
        plot_vrect(fig, anomaly, 'anomaly')
    # plot detect area if possible
    if detect is not None and max(detect):
        plot_vrect(fig, detect, 'detect')
    if height is not None:
        fig.update_layout(height=n_subplots * height)
    fig.show()
