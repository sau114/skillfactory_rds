from typing import Optional

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
        self.scaler = None  # transform data in same dimension
        pass

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

    def _scaler_partial_fit(self, data: pd.DataFrame) -> None:
        # partial fit
        if self.scaler is not None:
            self.scaler.partial_fit(data.values)
        return

    def _scaler_transform(self, data: pd.DataFrame) -> np.ndarray:
        # transform data
        if self.scaler is None:
            return data.values
        return self.scaler.transform(data.values)

    def _scaler_inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # inverse transform data
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)

    def _prefit(self,
                data: np.ndarray,
                **kwargs,
                ) -> None:
        # fit transformers
        # fit feature generators
        return

    def prefit(self,
               data_batch: pd.DataFrame,
               **kwargs,
               ) -> None:
        self._check_compliance(data_batch)
        self._scaler_partial_fit(data_batch)
        data_scaled = self._scaler_transform(data_batch)
        self._prefit(data_scaled, **kwargs)
        return

    def _partial_fit(self,
                     data: np.ndarray,
                     **kwargs,
                     ) -> None:
        # transform data
        # generate features
        # fit forecasters
        # store limits
        return

    def partial_fit(self,
                    data_batch: pd.DataFrame,
                    **kwargs,
                    ) -> None:
        self._check_compliance(data_batch)
        data_scaled = self._scaler_transform(data_batch)
        self._partial_fit(data_scaled, **kwargs)
        pass

    def _predict(self,
                 data: np.ndarray,
                 **kwargs,
                 ) -> np.ndarray:
        # transform data
        # generate features
        # use forecasters
        # check limits
        result = np.zeros_like(data, dtype=np.uint8)  # dummy predict
        return result

    def predict(self,
                data_batch: pd.DataFrame,
                reduce: bool = False,
                **kwargs,
                ) -> pd.Series:
        # common
        self._check_compliance(data_batch)
        data_scaled = self._scaler_transform(data_batch)
        result = pd.DataFrame(index=data_batch.index, data=self._predict(data_scaled, **kwargs), dtype='uint8')
        if result.shape[1] == data_batch.shape[1]:  # same number of columns
            result.columns = data_batch.columns
        if reduce:
            result = result.any(axis=1)
        return result


class DirectLimitWatchman(Watchman):
    # Checking limits of features directly

    def _init(self, **kwargs):
        self.scaler = None  # scaler is not used
        pass

    def _prefit(self,
                data: np.ndarray,
                **kwargs,
                ) -> None:
        # nothing
        return

    def _partial_fit(self,
                     data: np.ndarray,
                     **kwargs,
                     ) -> None:
        # store features limits
        data_min = data.min(axis=0, initial=None)
        data_max = data.max(axis=0, initial=None)
        if self.limits:
            self.limits['lo'] = np.fmin(self.limits['lo'], data_min)
            self.limits['hi'] = np.fmax(self.limits['hi'], data_max)
        else:
            # initialize empty dict by min and max features
            self.limits['lo'] = data_min
            self.limits['hi'] = data_max
        return

    def _predict(self,
                 data: np.ndarray,
                 tolerance: float = 0.02,
                 **kwargs,
                 ) -> np.ndarray:
        # check features limits with tolerance
        limits_center = (self.limits['hi'] + self.limits['lo']) / 2
        limits_scope = (self.limits['hi'] - self.limits['lo']) / 2
        limits_hi = limits_center + limits_scope * (1 + tolerance)
        limits_lo = limits_center - limits_scope * (1 + tolerance)
        result = (data < limits_lo) | (data > limits_hi)
        return result


class PcaLimitWatchman(Watchman):
    # Checking limits of features in principal components space
    # Also check high limit of presentation mean squared error (PMSE)

    def _init(self,
              n_components: int = 3,
              **kwargs):
        self.scaler = StandardScaler()  # preparing before pca
        self.transformer = IncrementalPCA(n_components=n_components)
        pass

    def explain_transformer(self) -> pd.Series:
        # scree of PCA for selecting the number of components
        scree = pd.Series(index=range(1, self.transformer.n_components_ + 1),
                          data=self.transformer.explained_variance_ratio_,
                          )
        return scree

    def _prefit(self,
                data: np.ndarray,
                **kwargs,
                ) -> None:
        self.transformer.partial_fit(data)
        return

    def _partial_fit(self,
                     data: np.ndarray,
                     **kwargs,
                     ) -> None:
        # transform to another space
        data_t = self.transformer.transform(data)
        # store features limits in new space
        data_t_min = data_t.min(axis=0, initial=None)
        data_t_max = data_t.max(axis=0, initial=None)
        # compute only high limit of PMSE, because data is scaled
        data_r = self.transformer.inverse_transform(data_t)  # restored data
        pmse_max = ((data - data_r) ** 2).mean(axis=1).max()
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
                 data: np.ndarray,
                 tolerance: float = 0.02,
                 **kwargs,
                 ) -> np.ndarray:
        # transform to another space
        data_t = self.transformer.transform(data)
        # compute PMSE
        data_r = self.transformer.inverse_transform(data_t)  # restored data
        pmse = ((data - data_r) ** 2).mean(axis=1)
        # check features limits with tolerance
        limits_center = (self.limits['hi'] + self.limits['lo']) / 2
        limits_scope = (self.limits['hi'] - self.limits['lo']) / 2
        limits_hi = limits_center + limits_scope * (1 + tolerance)
        limits_lo = limits_center - limits_scope * (1 + tolerance)
        result_f = (data_t < limits_lo) | (data_t > limits_hi)  # 2D-array [RxC]
        result_e = (pmse > self.limits['pmse'])[:, None]  # 2D-array [Rx1]
        result = np.hstack((result_f, result_e))
        return result


class IsolatingWatchman:
    # using Isolation Forest for anomaly detection

    def __init__(self,
                 max_trees: int = 1000,
                 max_samples: int = 256,
                 max_features: float = 1.0,
                 random_state: Optional[int] = None,
                 # contamination: Union[str, float] = 'auto',
                 generate_stat_features: bool = True,
                 ):
        self.forest = IsolationForest(n_estimators=0,
                                      max_samples=max_samples,
                                      max_features=max_features,
                                      random_state=random_state,
                                      # contamination=contamination,
                                      n_jobs=-1,
                                      warm_start=True,
                                      )
        self.max_trees = max_trees
        self.generate_stat_features = generate_stat_features
        return

    @staticmethod
    def _generate_stat_features(data: pd.DataFrame, window: int = 15) -> pd.DataFrame:
        float_data = data.select_dtypes(include='float')

        data_mean = float_data.rolling(window, min_periods=1).mean()
        data_mean.columns += '_mean'

        data_median = float_data.rolling(window, min_periods=1).median()
        data_median.columns += '_median'

        data_std = float_data.rolling(window, min_periods=1).std().fillna(0)
        data_std.columns += '_std'

        data_kurt = float_data.rolling(window, min_periods=1).kurt().fillna(0)
        data_kurt.columns += '_kurt'

        return pd.concat([data, data_mean, data_median, data_std, data_kurt], axis=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_trees={self.forest.n_estimators})'

    def prefit(self, data: pd.DataFrame) -> None:
        # nothing
        return

    def partial_fit(self, data: pd.DataFrame, increment: Optional[int] = 1) -> None:
        if increment is None or increment <= 0:
            inc = data.shape[0] // self.forest.max_samples
        else:
            inc = increment
        if self.generate_stat_features:
            data = self._generate_stat_features(data)
        self.forest.n_estimators = min(self.max_trees,
                                       self.forest.n_estimators + inc
                                       )
        self.forest.fit(data.values)
        return

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.generate_stat_features:
            data = self._generate_stat_features(data)
        result = (pd.Series(index=data.index, data=self.forest.predict(data.values))
                  .replace({1: 0, -1: 1})
                  .astype('uint8')
                  )
        return result


class LinearPredictWatchman:
    # using linear regressors for predict next value and calc limits of error

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
                # min(self.max_trees,
                #                                   self.regressors[c].n_estimators + increment
                #                                   )
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
            # self.regressors[c].n_estimators = self.regressors[c].n_estimators + increment
                # min(self.max_trees,
                #                                   self.regressors[c].n_estimators + increment
                #                                   )
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
