import random
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDRegressor


class WatchmanError(ValueError):
    pass


class LimitWatchman:
    # On learn - store limit values with tolerance.
    # On examine - values must be in limits.

    def __init__(self, ewma: Optional[str] = None):
        self.limits = None
        self.ewma = ewma
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(ewma={self.ewma})'

    def partial_fit(self,
                    data: pd.DataFrame,
                    tolerance: float = 0.05) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        if self.limits is None:
            self.limits = pd.DataFrame(index=data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = data.min()
            self.limits['hi'] = data.max()
        elif not self.limits.index.equals(data.columns):
            raise WatchmanError('Fields of limits is not equals to data columns')
        if self.ewma is not None:
            data = data.ewm(halflife=self.ewma, times=data.index.values).mean()
        mean = (data.max() + data.min()) / 2
        half_range = (data.max() - data.min()) / 2
        self.limits['lo'] = self.limits['lo'].combine(mean - half_range * (1 + tolerance), min)
        self.limits['hi'] = self.limits['hi'].combine(mean + half_range * (1 + tolerance), max)
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        if self.ewma is not None:
            data = data.ewm(halflife=self.ewma, times=data.index.values).mean()
        result = (data < self.limits['lo']) | (data > self.limits['hi'])
        return result


class LimitPcaWatchman:
    # Same as LimitWatchman, but in space of principal components.

    def __init__(self, n_components: int):
        self.limits = None
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=self.n_components)
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_components={self.n_components})'

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        self.scaler.partial_fit(data)
        return

    def partial_fit_pca(self, data: pd.DataFrame) -> None:
        self.pca.partial_fit(self.scaler.transform(data))
        return

    def get_pca_scree(self) -> pd.Series:
        scree = pd.Series(index=range(1, self.pca.n_components_ + 1),
                          data=self.pca.explained_variance_ratio_,
                          )
        return scree

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.01) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        pc_names = tuple('pc' + str(i) for i in range(self.pca.n_components))
        pc_data = pd.DataFrame(index=data.index,
                               columns=pc_names,
                               data=self.pca.transform(self.scaler.transform(data)),
                               )
        if self.limits is None:
            self.limits = pd.DataFrame(index=pc_data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = pc_data.min()
            self.limits['hi'] = pc_data.max()
        elif not self.limits.index.equals(pc_data.columns):
            raise WatchmanError('Fields of limits is not equals to data columns')
        mean = (pc_data.max() + pc_data.min()) / 2
        half_range = (pc_data.max() - pc_data.min()) / 2
        self.limits['lo'] = self.limits['lo'].combine(mean - half_range * (1 + tolerance), min)
        self.limits['hi'] = self.limits['hi'].combine(mean + half_range * (1 + tolerance), max)
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        pc_names = tuple('pc' + str(i) for i in range(self.pca.n_components))
        pc_data = pd.DataFrame(index=data.index,
                               columns=pc_names,
                               data=self.pca.transform(self.scaler.transform(data)),
                               )
        result = (pc_data < self.limits['lo']) | (pc_data > self.limits['hi'])
        return result


class SpePcaWatchman:
    # Same as LimitPcaWatchman, but watch only for square prediction error (SPE) aka Q statistic.

    def __init__(self, n_components: int):
        self.limits = None
        self.scaler = StandardScaler()
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=self.n_components)
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_components={self.n_components})'

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        self.scaler.partial_fit(data)
        return

    def partial_fit_pca(self, data: pd.DataFrame) -> None:
        self.pca.partial_fit(self.scaler.transform(data))
        return

    def get_pca_scree(self) -> pd.Series:
        scree = pd.Series(index=range(1, self.pca.n_components_ + 1),
                          data=self.pca.explained_variance_ratio_,
                          )
        return scree

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.05) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        scaled_data = pd.DataFrame(index=data.index,
                                   columns=data.columns,
                                   data=self.scaler.transform(data),
                                   )
        restored_data = pd.DataFrame(index=data.index,
                                     columns=data.columns,
                                     data=self.pca.inverse_transform(self.pca.transform(scaled_data.values)),
                                     )
        spe = ((scaled_data - restored_data) ** 2).mean(axis=1)
        if self.limits is None:
            self.limits = {
                'lo': spe.min(),
                'hi': spe.max(),
            }
        mean = (spe.max() + spe.min()) / 2
        half_range = (spe.max() - spe.min()) / 2
        self.limits['lo'] = min(self.limits['lo'], mean - half_range * (1 + tolerance))
        self.limits['hi'] = max(self.limits['hi'], mean + half_range * (1 + tolerance))
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        scaled_data = pd.DataFrame(index=data.index,
                                   columns=data.columns,
                                   data=self.scaler.transform(data),
                                   )
        restored_data = pd.DataFrame(index=data.index,
                                     columns=data.columns,
                                     data=self.pca.inverse_transform(self.pca.transform(scaled_data.values)),
                                     )
        spe = ((scaled_data - restored_data) ** 2).mean(axis=1)
        result = (spe < self.limits['lo']) | (spe > self.limits['hi'])
        result = result.astype('uint8')
        return result


class IsolatingWatchman:
    # using Isolation Forest for anomaly detection

    def __init__(self,
                 max_trees: int = 1000,
                 max_samples: int = 256,
                 max_features: float = 1.0,
                 random_state: Optional[int] = None
                 ):
        self.forest = IsolationForest(n_estimators=0,
                                      max_samples=max_samples,
                                      max_features=max_features,
                                      random_state=random_state,
                                      contamination='auto',
                                      n_jobs=-1,
                                      warm_start=True,
                                      )
        self.max_trees = max_trees
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_trees={self.forest.n_estimators})'

    def partial_fit(self, data: pd.DataFrame, increment: Optional[int] = 1) -> None:
        if increment is None or increment <= 0:
            inc = data.shape[0] // self.forest.max_samples
        else:
            inc = increment
        self.forest.n_estimators = min(self.max_trees,
                                       self.forest.n_estimators + inc
                                       )
        self.forest.fit(data)
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        result = (pd.Series(index=data.index, data=self.forest.predict(data))
                  .replace({1: 0, -1: 1})
                  .astype('uint8')
                  )
        return result


class LinearPredictWatchman:
    # using linear regressors for predict next value and calc limits of error

    def __init__(self,
                 random_state: Optional[int] = None,
                 also_compute_spe: bool = True,
                 ):
        self.limits = None  # predict error limits
        self.scaler = StandardScaler()
        self.regressors = None
        self.random_state = random_state
        if also_compute_spe:
            self.spe_hi = 0.0
        else:
            self.spe_hi = None
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_features={len(self.regressors)})'

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        if self.regressors is None:
            # create regressors
            self.regressors = {c: SGDRegressor(random_state=self.random_state,
                                               warm_start=True,
                                               ) for c in data.columns}
        if self.limits is None:
            # create limits
            self.limits = pd.DataFrame(index=data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = 0.0
            self.limits['hi'] = 0.0
        self.scaler.partial_fit(data)
        return

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.05) -> None:
        # fit regressors on data
        x_ = self.scaler.transform(data)
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            # self.regressors[c].fit(x, y)
            self.regressors[c].partial_fit(x, y)
            # compute predict errors
            errors[c] = y - self.regressors[c].predict(x)
        # update limits
        center = (errors.max() + errors.min()) / 2
        scope = (errors.max() - errors.min()) / 2
        self.limits['lo'] = self.limits['lo'].combine(center - scope * (1 + tolerance), min)
        self.limits['hi'] = self.limits['hi'].combine(center + scope * (1 + tolerance), max)
        if self.spe_hi is not None:
            self.spe_hi = max((errors**2).mean(axis=1).max() * (1 + tolerance), self.spe_hi)
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # predict values and check error's limits
        result = pd.DataFrame(index=data.index, columns=data.columns, data=0, dtype='uint8')
        x_ = self.scaler.transform(data)
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            errors[c] = y - self.regressors[c].predict(x)
        result.loc[data.index[1]:] = (errors < self.limits['lo']) | (errors > self.limits['hi'])
        if self.spe_hi is not None:
            spe = (errors**2).mean(axis=1)
            result['spe'] = 0
            result.loc[data.index[1]:, 'spe'] = spe > self.spe_hi
        result = result.astype('uint8')
        return result
