from typing import Union

import pandas as pd


def time_span_metrics(y_true: Union[pd.Series, pd.DataFrame],
                      y_pred: Union[pd.Series, pd.DataFrame],
                      time_span: str = '15 min',
                      beta: float = 1.0,
                      ) -> tuple:
    # collapse to bool vector
    if isinstance(y_true, pd.DataFrame):
        true_vec = (y_true != 0).any(axis=1)
    else:
        true_vec = (y_true != 0)
    if isinstance(y_pred, pd.DataFrame):
        pred_vec = (y_pred != 0).any(axis=1)
    else:
        pred_vec = (y_pred != 0)
    # collapse by time span
    true_vec = true_vec.resample(time_span, origin='start_day').max()
    pred_vec = pred_vec.resample(time_span, origin='start_day').max()
    # calc hits (always >= 0)
    tp = (true_vec & pred_vec).sum()
    fp = (~true_vec & pred_vec).sum()
    fn = (true_vec & ~pred_vec).sum()
    # precision
    if tp + fp:
        precision_score = tp / (tp + fp)
    else:
        # undefined
        precision_score = float('nan')
    # recall
    if tp + fn:
        recall_score = tp / (tp + fn)
    else:
        # undefined
        recall_score = float('nan')
    # f_beta (default f1)
    if tp + fp + fn:
        f_beta_score = (1+beta**2) * tp / ((1+beta**2) * tp + beta**2 * fn + fp)
    else:
        # undefined
        f_beta_score = float('nan')
    return precision_score, recall_score, f_beta_score
