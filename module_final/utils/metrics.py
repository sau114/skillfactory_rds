from typing import Union

import pandas as pd


def precision(y_true: Union[pd.Series, pd.DataFrame],
              y_pred: Union[pd.Series, pd.DataFrame],
              batch_period: str = '15 min',
              ) -> float:
    # collapse to vector
    if isinstance(y_true, pd.DataFrame):
        true_vec = (y_true != 0).any(axis=1)
    else:
        true_vec = (y_true != 0)
    if isinstance(y_pred, pd.DataFrame):
        pred_vec = (y_pred != 0).any(axis=1)
    else:
        pred_vec = (y_pred != 0)
    # collapse by period, like any in period
    true_vec = true_vec.resample(batch_period, origin='start_day').max()
    pred_vec = pred_vec.resample(batch_period, origin='start_day').max()
    # calc score
    if ~pred_vec.any():
        # metric is undefined
        return float('nan')
    tp = (true_vec & pred_vec).sum()
    fp = (~true_vec & pred_vec).sum()
    score = tp / (tp + fp)
    return score


def recall(y_true: Union[pd.Series, pd.DataFrame],
           y_pred: Union[pd.Series, pd.DataFrame],
           batch_period: str = '15 min',
           ) -> float:
    # collapse to vector
    if isinstance(y_true, pd.DataFrame):
        true_vec = (y_true != 0).any(axis=1)
    else:
        true_vec = (y_true != 0)
    if isinstance(y_pred, pd.DataFrame):
        pred_vec = (y_pred != 0).any(axis=1)
    else:
        pred_vec = (y_pred != 0)
    # collapse by period, like any in period
    true_vec = true_vec.resample(batch_period, origin='start_day').max()
    pred_vec = pred_vec.resample(batch_period, origin='start_day').max()
    # calc score
    if ~true_vec.any():
        # metric is undefined
        return float('nan')
    tp = (true_vec & pred_vec).sum()
    fn = (true_vec & ~pred_vec).sum()
    score = tp / (tp + fn)
    return score


def f1_score(y_true: Union[pd.Series, pd.DataFrame],
             y_pred: Union[pd.Series, pd.DataFrame],
             batch_period: str = '15 min',
             ) -> float:
    # collapse to vector
    if isinstance(y_true, pd.DataFrame):
        true_vec = (y_true != 0).any(axis=1)
    else:
        true_vec = (y_true != 0)
    if isinstance(y_pred, pd.DataFrame):
        pred_vec = (y_pred != 0).any(axis=1)
    else:
        pred_vec = (y_pred != 0)
    # collapse by period, like any in period
    true_vec = true_vec.resample(batch_period, origin='start_day').max()
    pred_vec = pred_vec.resample(batch_period, origin='start_day').max()
    # calc score
    if ~true_vec.any() & ~pred_vec.any():
        # metric is undefined
        return float('nan')
    tp = (true_vec & pred_vec).sum()
    fp = (~true_vec & pred_vec).sum()
    fn = (true_vec & ~pred_vec).sum()
    if tp + fp:
        pre = tp / (tp + fp)
    else:
        pre = float('nan')
    if tp + fn:
        rec = tp / (tp + fn)
    else:
        rec = float('nan')
    if pd.isna(pre) | pd.isna(rec):
        score = 0.
    else:
        score = 2 * pre * rec / (pre + rec)
    return score
