from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor, LGBMClassifier


class WatchmanError(ValueError):
    pass


class Watchman:
    # root class of Watchmen

    def _init(self, **kwargs):
        self.scaler = StandardScaler()  # transform data in same dimension
        return

    def __init__(self,
                 random_state: Optional[int] = None,
                 **kwargs,
                 ):
        self.data_dtypes = None  # names and types of data features
        self.limits = dict()  # all limits for predict anomalies
        self.random_state = random_state  # random seed for everybody
        self._init(**kwargs)
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_features={self.data_dtypes.shape[0]})'

    def _check_compliance(self,
                          data_batch: pd.DataFrame,
                          ) -> None:
        if self.data_dtypes is None:
            self.data_dtypes = data_batch.dtypes
            return
        if not data_batch.dtypes.equals(self.data_dtypes):
            raise WatchmanError('Batch is not correspond previous data')
        return

    def _prefit(self,
                data: pd.DataFrame,
                **kwargs,
                ) -> None:
        self.scaler.partial_fit(data)
        data_s = self.scaler.transform(data)
        # prefit
        return

    def prefit(self,
               data_batch: pd.DataFrame,
               **kwargs,
               ) -> None:
        self._check_compliance(data_batch)
        self._prefit(data_batch, **kwargs)
        return

    def _partial_fit(self,
                     data: pd.DataFrame,
                     **kwargs,
                     ) -> None:
        data_s = self.scaler.transform(data)
        # partial fit
        return

    def partial_fit(self,
                    data_batch: pd.DataFrame,
                    **kwargs,
                    ) -> None:
        self._check_compliance(data_batch)
        self._partial_fit(data_batch, **kwargs)
        return

    def _predict(self,
                 data: pd.DataFrame,
                 **kwargs,
                 ) -> pd.DataFrame:
        data_s = self.scaler.transform(data)
        # predict
        result = pd.DataFrame(index=data.index,
                              columns=data.columns,
                              data=0,
                              dtype='uint8',
                              )
        return result

    def predict(self,
                data_batch: pd.DataFrame,
                reduce: bool = False,
                **kwargs,
                ) -> Union[pd.DataFrame, pd.Series]:
        # common
        self._check_compliance(data_batch)
        result = self._predict(data_batch, **kwargs)
        if reduce:
            result = result.any(axis=1)
        return result


class DirectLimitWatchman(Watchman):
    # Checking limits of features directly

    def _init(self, **kwargs):
        # without scaler
        return

    def _prefit(self,
                data: pd.DataFrame,
                **kwargs,
                ) -> None:
        # nothing
        return

    def _partial_fit(self,
                     data: pd.DataFrame,
                     **kwargs,
                     ) -> None:
        data_v = data.values
        # store features limits
        data_v_min = data_v.min(axis=0, initial=None)
        data_v_max = data_v.max(axis=0, initial=None)
        if self.limits:
            self.limits['lo'] = np.fmin(self.limits['lo'], data_v_min)
            self.limits['hi'] = np.fmax(self.limits['hi'], data_v_max)
        else:
            # initialize empty dict by min and max features
            self.limits['lo'] = data_v_min
            self.limits['hi'] = data_v_max
        return

    def _predict(self,
                 data: pd.DataFrame,
                 tolerance: float = 0.02,
                 **kwargs,
                 ) -> pd.DataFrame:
        data_v = data.values
        # check features limits with tolerance
        limits_center = (self.limits['hi'] + self.limits['lo']) / 2
        limits_scope = (self.limits['hi'] - self.limits['lo']) / 2
        limits_hi = limits_center + limits_scope * (1 + tolerance)
        limits_lo = limits_center - limits_scope * (1 + tolerance)
        # summarize
        result = pd.DataFrame(index=data.index,
                              columns=data.columns,
                              data=(data_v < limits_lo) | (data_v > limits_hi),
                              dtype='uint8',
                              )
        return result


class PcaLimitWatchman(Watchman):
    # Checking limits of features in principal components space
    # Also check high limit of presentation mean squared error (PMSE)

    def _init(self,
              n_components: int = 3,
              **kwargs):
        self.scaler = StandardScaler()  # preparing before pca
        assert 0 < n_components, 'Number of components must be greater than zero'
        self.transformer = IncrementalPCA(n_components=n_components)
        return

    def explain_transformer(self) -> pd.Series:
        # scree of PCA for selecting the number of components
        scree = pd.Series(index=range(1, self.transformer.n_components_ + 1),
                          data=self.transformer.explained_variance_ratio_,
                          )
        return scree

    def _prefit(self,
                data: pd.DataFrame,
                **kwargs,
                ) -> None:
        self.scaler.partial_fit(data)
        data_s = self.scaler.transform(data)
        self.transformer.partial_fit(data_s)
        return

    @staticmethod
    def _mean_squared_error(data_1: np.ndarray,
                            data_2: np.ndarray,
                            ) -> np.ndarray:
        return ((data_1 - data_2) ** 2).mean(axis=1)

    def _partial_fit(self,
                     data: pd.DataFrame,
                     **kwargs,
                     ) -> None:
        data_s = self.scaler.transform(data)
        # transform to another space
        data_t = self.transformer.transform(data_s)
        # store features limits in new space
        data_t_min = data_t.min(axis=0, initial=None)
        data_t_max = data_t.max(axis=0, initial=None)
        # compute only high limit of PMSE, because data is scaled
        data_r = self.transformer.inverse_transform(data_t)  # restored data
        pmse_max = self._mean_squared_error(data_s, data_r).max(initial=None)
        if self.limits:
            self.limits['lo'] = np.fmin(self.limits['lo'], data_t_min)
            self.limits['hi'] = np.fmax(self.limits['hi'], data_t_max)
            self.limits['pmse'] = max(self.limits['pmse'], pmse_max)
        else:
            # initialize empty dict by min and max features
            self.limits['lo'] = data_t_min
            self.limits['hi'] = data_t_max
            self.limits['pmse'] = pmse_max
        return

    def _predict(self,
                 data: pd.DataFrame,
                 tolerance: float = 0.02,
                 **kwargs,
                 ) -> pd.DataFrame:
        data_s = self.scaler.transform(data)
        # transform to another space
        data_t = self.transformer.transform(data_s)
        # compute PMSE
        data_r = self.transformer.inverse_transform(data_t)  # restored data
        pmse = self._mean_squared_error(data_s, data_r)
        # check features limits with tolerance
        limits_center = (self.limits['hi'] + self.limits['lo']) / 2
        limits_scope = (self.limits['hi'] - self.limits['lo']) / 2
        limits_hi = limits_center + limits_scope * (1 + tolerance)
        limits_lo = limits_center - limits_scope * (1 + tolerance)
        # summarize
        result_f = (data_t < limits_lo) | (data_t > limits_hi)  # 2D-array [RxC]
        result_e = (pmse > self.limits['pmse'])[:, None]  # 2D-array [Rx1]
        result = pd.DataFrame(index=data.index,
                              data=np.hstack((result_f, result_e)),
                              dtype='uint8',
                              )
        return result


class IsoForestWatchman(Watchman):
    # Isolating Forest algorithm for time series

    def _init(self,
              generate_features: bool = True,
              window_sparsity: float = 0.5,
              window_overlap: float = 0.1,
              max_trees: int = 100,
              **kwargs):
        self.forest = IsolationForest(
            n_estimators=0,
            max_samples='auto',
            contamination='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=self.random_state,
            warm_start=True,
        )
        self.generate_features = generate_features
        self.float_features = []
        assert 0.0 < window_sparsity <= 1.0, 'Sparsity must be in (0.0; 1.0]'
        self.window_sparsity = window_sparsity
        assert 0.0 <= window_overlap < 1.0, 'Overlap must be in [0.0; 1.0)'
        self.window_overlap = window_overlap
        assert 1 <= max_trees <= 10_000, 'Incorrect maximum number of trees'
        self.max_trees = max_trees
        self._samples_per_tree = 256  # default number of samples per tree
        self._batches_len = []  # save length of batches for counting n_trees
        return

    def _compute_window_size_shift(self):
        size = round(self._samples_per_tree / self.window_sparsity)  # [samples_per_tree; +Inf)
        shift = round((size - 1) * (1 - self.window_overlap)) + 1  # [1; samples_per_window]
        return size, shift

    def _estimate_n_trees(self) -> int:
        size, shift = self._compute_window_size_shift()
        return sum(int(np.ceil(max(b_len - size, 0) / shift)) + 1 for b_len in self._batches_len)

    def _prefit(self,
                data: pd.DataFrame,
                **kwargs,
                ) -> None:
        self._batches_len.append(data.shape[0])
        # at least one tree per batch
        self.max_trees = max(self.max_trees, len(self._batches_len))
        # if we need too much trees, increase sparsity
        while self._estimate_n_trees() > self.max_trees:
            self.window_sparsity *= 0.9
        return

    @staticmethod
    def _generate_features(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        data_f = data.select_dtypes(include='float')
        # mean
        data_mean = data_f.rolling(window, min_periods=1).mean()
        data_mean.columns = 'mean_' + data_mean.columns
        # median
        data_median = data_f.rolling(window, min_periods=1).median()
        data_mean.columns = 'median_' + data_mean.columns
        # standard deviation
        data_std = data_f.rolling(window, min_periods=1).std().fillna(0)
        data_mean.columns = 'std_' + data_mean.columns
        # kurtosis
        data_kurt = data_f.rolling(window, min_periods=1).kurt().fillna(0)
        data_mean.columns = 'kurt_' + data_mean.columns
        return pd.concat([data_mean, data_median, data_std, data_kurt], axis=1)

    def _partial_fit(self,
                     data: pd.DataFrame,
                     **kwargs,
                     ) -> None:
        if self.generate_features:
            data = pd.concat([data, self._generate_features(data)], axis=1)
        size, shift = self._compute_window_size_shift()
        for i in range(0, data.shape[0], shift):
            self.forest.n_estimators += 1  # one tree per window
            self.forest.fit(data.iloc[i:i+size].values)
            # if right border of window is out of data
            if i+size >= data.shape[0]:
                break
        return

    def _predict(self,
                 data: pd.DataFrame,
                 **kwargs,
                 ) -> pd.DataFrame:
        if self.generate_features:
            data = pd.concat([data, self._generate_features(data)], axis=1)
        result = pd.DataFrame(index=data.index,
                              data=(1-self.forest.predict(data.values)) // 2,  # replace (1, -1) to (0, 1)
                              dtype='uint8',
                              )
        return result


class LinearPredictWatchman(Watchman):
    # using linear models (SGDRegressor and SGDClassifier) for predict next value and calc limits of error

    def _init(self,
              split_by_types: bool = True,
              **kwargs):
        self.scaler = StandardScaler()  # preparing for SGDRegressor
        self.forecasters = dict()
        self.split_by_types = split_by_types
        return


class LinPredictWatchman:

    def __init__(self,
                 random_state: Optional[int] = None,
                 also_compute_spe: bool = True,
                 use_log_state: bool = True,
                 ):
        self.limits = None  # predict error limits
        self.scaler = StandardScaler()
        self.regressors = None
        self.random_state = random_state
        if also_compute_spe:
            self.spe_hi = 0.0
        else:
            self.spe_hi = None
        self.use_log = use_log_state
        self.lin_columns = None
        self.log_columns = None
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_features={len(self.regressors)})'

    def prefit(self, data: pd.DataFrame) -> None:
        # create regressors
        if self.regressors is None:
            if self.use_log:
                self.log_columns = [c for c in data.columns if c.endswith('_state')]
                self.lin_columns = [c for c in data.columns if not c.endswith('_state')]
            else:
                self.lin_columns = [c for c in data.columns]
            self.regressors = dict()
            for c in data.columns:
                if c in self.lin_columns:
                    self.regressors[c] = SGDRegressor(random_state=self.random_state,
                                                      warm_start=True,
                                                      # can be: squared_error, huber, ...
                                                      loss='squared_error',
                                                      )
                else:
                    self.regressors[c] = SGDClassifier(random_state=self.random_state,
                                                       warm_start=True,
                                                       # can be: hinge, log/log_loss, squared_hinge, perceptron, ...
                                                       loss='log',
                                                       )
        # create limits
        if self.limits is None:
            self.limits = pd.DataFrame(index=data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = 0.0
            self.limits['hi'] = 0.0
        # fit scaler
        self.scaler.partial_fit(data[self.lin_columns])
        return

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.05) -> None:
        # fit regressors on data
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            # self.regressors[c].fit(x, y)
            if c in self.lin_columns:
                self.regressors[c].partial_fit(x, y)
            else:
                self.regressors[c].partial_fit(x, y, classes=np.array([0, 1, 2]))
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
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            errors[c] = y - self.regressors[c].predict(x)
        result.loc[data.index[1]:] = (errors < self.limits['lo']) | (errors > self.limits['hi'])
        if self.spe_hi is not None:
            spe = (errors**2).mean(axis=1)
            result = result.assign(spe=0.0)
            result.loc[data.index[1]:, 'spe'] = spe > self.spe_hi
        result = result.astype('uint8')
        return result


class ForestPredictWatchman:
    # using random forest regressors for predict next value and calc limits of error

    def __init__(self,
                 random_state: Optional[int] = None,
                 also_compute_spe: bool = True,
                 use_log_state: bool = False,
                 max_trees: int = 100,
                 ):
        self.limits = None  # predict error limits
        self.scaler = StandardScaler()  # need for spe
        self.regressors = None
        self.random_state = random_state
        if also_compute_spe:
            self.spe_hi = 0.0
        else:
            self.spe_hi = None
        self.use_log = use_log_state
        self.lin_columns = None
        self.log_columns = None
        self.max_trees = max_trees
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_features={len(self.regressors)})'

    def prefit(self, data: pd.DataFrame) -> None:
        # create regressors
        if self.regressors is None:
            if self.use_log:
                self.log_columns = [c for c in data.columns if c.endswith('_state')]
                self.lin_columns = [c for c in data.columns if not c.endswith('_state')]
            else:
                self.lin_columns = [c for c in data.columns]
            self.regressors = dict()
            for c in data.columns:
                if c in self.lin_columns:
                    self.regressors[c] = RandomForestRegressor(
                        n_estimators=0,
                        criterion='squared_error',
                        random_state=self.random_state,
                        n_jobs=-1,
                        warm_start=True,
                    )
                else:
                    self.regressors[c] = RandomForestClassifier(
                        n_estimators=0,
                        criterion='gini',
                        random_state=self.random_state,
                        n_jobs=-1,
                        warm_start=True,
                    )
        # create limits
        if self.limits is None:
            self.limits = pd.DataFrame(index=data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = 0.0
            self.limits['hi'] = 0.0
        # fit scaler
        self.scaler.partial_fit(data[self.lin_columns])
        return

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.05, increment: int = 3) -> None:
        # fit regressors on data
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            self.regressors[c].n_estimators = self.regressors[c].n_estimators + increment
            if c in self.lin_columns:
                self.regressors[c].fit(x, y)
            else:
                self.regressors[c].fit(x, y)
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
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            errors[c] = y - self.regressors[c].predict(x)
        result.loc[data.index[1]:] = (errors < self.limits['lo']) | (errors > self.limits['hi'])
        if self.spe_hi is not None:
            spe = (errors**2).mean(axis=1)
            result = result.assign(spe=0.0)
            result.loc[data.index[1]:, 'spe'] = spe > self.spe_hi
        result = result.astype('uint8')
        return result


class GbmPredictWatchman:
    # using gradient boosting regressors for predict next value and calc limits of error

    def __init__(self,
                 random_state: Optional[int] = None,
                 also_compute_spe: bool = True,
                 use_log_state: bool = True,
                 max_trees: int = 100,
                 ):
        self.limits = None  # predict error limits
        self.scaler = StandardScaler()  # need for spe
        self.regressors = None
        self.random_state = random_state
        if also_compute_spe:
            self.spe_hi = 0.0
        else:
            self.spe_hi = None
        self.use_log = use_log_state
        self.lin_columns = None
        self.log_columns = None
        self.max_trees = max_trees
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_features={len(self.regressors)})'

    def prefit(self, data: pd.DataFrame) -> None:
        # create regressors
        if self.regressors is None:
            if self.use_log:
                self.log_columns = [c for c in data.columns if c.endswith('_state')]
                self.lin_columns = [c for c in data.columns if not c.endswith('_state')]
            else:
                self.lin_columns = [c for c in data.columns]
            self.regressors = dict()
            for c in data.columns:
                if c in self.lin_columns:
                    self.regressors[c] = LGBMRegressor(
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                else:
                    self.regressors[c] = LGBMClassifier(
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
        # create limits
        if self.limits is None:
            self.limits = pd.DataFrame(index=data.columns, columns=['lo', 'hi'])
            self.limits['lo'] = 0.0
            self.limits['hi'] = 0.0
        # fit scaler
        self.scaler.partial_fit(data[self.lin_columns])
        return

    def partial_fit(self, data: pd.DataFrame, tolerance: float = 0.05, increment: int = 3) -> None:
        # fit regressors on data
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            if c in self.lin_columns:
                try:
                    self.regressors[c].best_iteration_
                except:
                    self.regressors[c].fit(x, y)
                else:
                    self.regressors[c].fit(x, y, init_model=self.regressors[c])
            else:
                try:
                    self.regressors[c].best_iteration_
                except:
                    self.regressors[c].fit(x, y)
                else:
                    self.regressors[c].fit(x, y, init_model=self.regressors[c])
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
        x_ = data.copy()
        x_.loc[:, self.lin_columns] = self.scaler.transform(data.loc[:, self.lin_columns])
        x_ = x_.values
        x = x_[:-1]  # without last row
        errors = pd.DataFrame(index=data.index[1:], columns=data.columns)
        for j, c in enumerate(data.columns):
            y = x_[1:, j]  # without first row
            errors[c] = y - self.regressors[c].predict(x)
        result.loc[data.index[1]:] = (errors < self.limits['lo']) | (errors > self.limits['hi'])
        if self.spe_hi is not None:
            spe = (errors**2).mean(axis=1)
            result = result.assign(spe=0.0)
            result.loc[data.index[1]:, 'spe'] = spe > self.spe_hi
        result = result.astype('uint8')
        return result
