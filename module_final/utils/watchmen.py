from typing import Optional, Union

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA


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

    def learn(self,
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

    def examine(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        if self.ewma is not None:
            data = data.ewm(halflife=self.ewma, times=data.index.values).mean()
        result = (data < self.limits['lo']) | (data > self.limits['hi'])
        return result


class LimitPcaWatchman:
    # Same as LimitWatchman, but in space of principal components.

    def __init__(self, n_components: Union[str, int] = 'auto'):
        self.limits = None
        self.scaler = StandardScaler()
        self.n_components = n_components
        if self.n_components == 'auto':
            self.pca = IncrementalPCA(n_components=1)
        elif isinstance(self.n_components, int) and self.n_components > 0:
            self.pca = IncrementalPCA(n_components=self.n_components)
        else:
            raise WatchmanError('Wrong value of n_components')
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_components={self.n_components})'

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        self.scaler.partial_fit(data)
        if self.n_components == 'auto':
            pca = PCA(n_components='mle')
            pca.fit(data)
            self.pca.n_components = max(self.pca.n_components, pca.n_components_)
        return

    def partial_fit_pca(self, data: pd.DataFrame) -> None:
        self.pca.partial_fit(self.scaler.transform(data))
        return

    def learn(self, data: pd.DataFrame, tolerance: float = 0.01) -> None:
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

    def examine(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def __init__(self, n_components: Union[str, int] = 'auto'):
        self.limits = None
        self.scaler = StandardScaler()
        self.n_components = n_components
        if self.n_components == 'auto':
            self.pca = IncrementalPCA(n_components=1)
        elif isinstance(self.n_components, int) and self.n_components > 0:
            self.pca = IncrementalPCA(n_components=self.n_components)
        else:
            raise WatchmanError('Wrong value of n_components')
        return

    def __repr__(self):
        return f'{self.__class__.__name__}(n_components={self.n_components})'

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        self.scaler.partial_fit(data)
        if self.n_components == 'auto':
            pca = PCA(n_components='mle')
            pca.fit(data)
            self.pca.n_components = max(self.pca.n_components, pca.n_components_)
        return

    def partial_fit_pca(self, data: pd.DataFrame) -> None:
        self.pca.partial_fit(self.scaler.transform(data))
        return

    def learn(self, data: pd.DataFrame, tolerance: float = 0.05) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        rest_data = pd.DataFrame(index=data.index,
                                 columns=data.columns,
                                 data=self.pca.inverse_transform(self.pca.transform(self.scaler.transform(data))),
                                 )
        spe = ((data - rest_data) ** 2).mean(axis=1)
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

    def examine(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        rest_data = pd.DataFrame(index=data.index,
                                 columns=data.columns,
                                 data=self.pca.inverse_transform(self.pca.transform(self.scaler.transform(data))),
                                 )
        spe = ((data - rest_data) ** 2).mean(axis=1)
        result = (spe < self.limits['lo']) | (spe > self.limits['hi'])
        return result
