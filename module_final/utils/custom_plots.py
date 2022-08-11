from typing import Optional

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_stacked(data: pd.DataFrame,
                 title: str = '',
                 suffixes: Optional[tuple] = None,
                 anomalies: Optional[pd.Series] = None,
                 detect: Optional[pd.Series] = None,
                 height: Optional[int] = None,
                 ) -> None:
    # Plotting time series from dataframe in stacked subplots style with shared time axe.
    # Time series can group by suffix.
    # Anomaly and detect intervals can be shown on subplots.

    def plot_vrect(figr: go.Figure,
                   series: pd.Series,
                   vrect_type: str,
                   ) -> None:
        # plot rectangle area for anomalies or detect
        if vrect_type == 'anomaly':
            prefix = 'A'
            common_kwargs = {
                'annotation_position': 'outside top left',
                'fillcolor':  'red',
                'opacity': 0.25,
            }
        elif vrect_type == 'detect':
            prefix = 'D'
            common_kwargs = {
                'annotation_position': 'outside bottom right',
                'fillcolor':  'green',
                'opacity': 0.25,
            }
        else:
            raise ValueError(f'Wrong vrect type {vrect_type}')
        vector = series.copy()
        vector[vector.index[-1] + vector.index.freq] = 0  # always finished by normal state
        state = 0
        period_kwargs = {'x0': None, 'x1': None, 'annotation_text': None}
        for i in vector.index:
            if vector[i] != state:
                # if this anomaly finished
                if state != 0:
                    figr.add_vrect(**period_kwargs,
                                   **common_kwargs,
                                   )
                # next anomaly starting
                state = vector[i]
                period_kwargs['x0'] = i
                period_kwargs['annotation_text'] = prefix + str(vector[i])
            period_kwargs['x1'] = i
        return

    # make figure with subplots
    if suffixes is not None:
        # columns outside suffixes will in additional subplot
        n_subplots = len(suffixes) + max(0 if c.endswith(suffixes) else 1 for c in data.columns)
    else:
        suffixes = ()
        n_subplots = 1
    fig = make_subplots(rows=n_subplots,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_titles=suffixes + ('other',),
                        # x_title=data.index.name,
                        )
    # fill subplots using suffixes
    for i_subplot, suffix in enumerate(suffixes):
        for k in [c for c in data.columns if c.endswith(suffix)]:
            fig.add_scatter(x=data[k].index,
                            y=data[k],
                            name=k,
                            row=i_subplot + 1,
                            col=1,
                            )
    # plot other in additional subplots
    for k in [c for c in data.columns if not c.endswith(suffixes)]:
        fig.add_scatter(x=data[k].index,
                        y=data[k],
                        name=k,
                        row=n_subplots,
                        col=1,
                        )
    # plot anomaly area if possible
    if anomalies is not None and max(anomalies):
        plot_vrect(fig, anomalies, 'anomaly')
    # plot detect area if possible
    if detect is not None and max(detect):
        plot_vrect(fig, detect, 'detect')
    if height is not None:
        fig.update_layout(height=n_subplots * height)
    fig.update_layout(title_text=title)
    fig.show()
