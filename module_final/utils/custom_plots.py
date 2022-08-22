from typing import Optional, Union

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adtk.data import to_events


def plot_stacked(data: pd.DataFrame,
                 # suffixes: Optional[tuple] = None,
                 group: str = 'value_unit',  # scheme of grouping curves
                 faults: Optional[Union[pd.Series, pd.DataFrame]] = None,
                 detect: Optional[Union[pd.Series, pd.DataFrame]] = None,
                 title: str = '',
                 height: Optional[int] = 200,
                 ) -> None:
    # Plotting time series from dataframe in stacked subplots style with shared time axe.
    # Time series can group, expected that named in scheme 'name_value_unit'.
    # Anomaly and detect intervals can be shown on subplots.

    def add_marks(fig: go.Figure,
                  events: list,
                  vrect_type: str,
                  row: Union[str, int] = 'all',
                  ) -> None:
        # plot rectangle area for anomalies or detect
        if vrect_type == 'fault':
            common_kwargs = {
                'fillcolor':  'red',
                'opacity': 0.3,
            }
        elif vrect_type == 'detect':
            common_kwargs = {
                'fillcolor':  'green',
                'opacity': 0.3,
            }
        else:
            raise ValueError(f'Unknown vrect type {vrect_type}')
        for e in events:
            if isinstance(e, tuple):
                fig.add_vrect(x0=e[0],
                              x1=e[1],
                              row=row,
                              **common_kwargs,
                              )
            else:
                fig.add_vline(x=e,
                              row=row,
                              **common_kwargs,
                              )
        return

    # generate grouping rule
    if group == 'unit':
        # using unit in name_value_unit
        endings = tuple(sorted(set(s.split('_')[-1] for s in data.columns)))
    elif group == 'value_unit':
        # using value_unit in name_value_unit
        endings = tuple(sorted(set('_'.join(s.split('_')[-2:]) for s in data.columns)))
    # custom grouping rules write here
    else:
        # not grouping
        endings = ('',)
    # destination subplot
    n_subplots = len(endings)
    subplot_number = dict()
    for i, end in enumerate(endings):
        for k in [c for c in data.columns if c.endswith(end)]:
            subplot_number[k] = i + 1
    # create figure
    fig = make_subplots(rows=n_subplots,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_titles=endings,
                        )
    # fill subplots using endings
    for c in data.columns:
        fig.add_scatter(x=data[c].index,
                        y=data[c],
                        name=c,
                        row=subplot_number[c],
                        col=1,
                        )
    # plot faults area if possible and needed
    if faults is not None and max(faults):
        if isinstance(faults, pd.Series):
            add_marks(fig, to_events(faults > 0), 'fault')
        else:
            for c in faults.columns:
                add_marks(fig, to_events(faults[c] > 0), 'fault', subplot_number[c])
    # plot detect area if possible and needed
    if detect is not None and max(detect):
        if isinstance(detect, pd.Series):
            add_marks(fig, to_events(detect > 0), 'detect')
        else:
            for c in detect.columns:
                add_marks(fig, to_events(detect[c] > 0), 'detect', subplot_number[c])
    fig.update_layout(height=n_subplots * height)
    fig.update_layout(title_text=title)
    fig.show()
