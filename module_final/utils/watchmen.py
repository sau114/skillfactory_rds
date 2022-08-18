from typing import Optional

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


class WatchmanError(ValueError):
    pass


class LimitWatchman:
    # On learn - store limit values with tolerance.
    # On examine - values must be in limits.

    def __init__(self, ewma: Optional[str] = None):
        self.limits = None
        self.halflife = ewma
        return

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
        if self.halflife is not None:
            data = data.ewm(halflife=self.halflife, times=data.index.values).mean()
        mean = (data.max() + data.min()) / 2
        half_range = (data.max() - data.min()) / 2
        self.limits['lo'] = self.limits['lo'].combine(mean - half_range * (1 + tolerance), min)
        self.limits['hi'] = self.limits['hi'].combine(mean + half_range * (1 + tolerance), max)
        return

    def examine(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        if self.halflife is not None:
            data = data.ewm(halflife=self.halflife, times=data.index.values).mean()
        result = (data < self.limits['lo']) | (data > self.limits['hi'])
        return result


class LimitPcaWatchman:
    # Same as LimitWatchman, but in space of principal components.

    def __init__(self, n_components: int):
        self.limits = {}  # dict of watched values
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=n_components)
        self._pc_names = tuple('pc'+str(i) for i in range(n_components))
        return

    def partial_fit_scaler(self, data: pd.DataFrame) -> None:
        self.scaler.partial_fit(data)
        return

    def partial_fit_pca(self, data: pd.DataFrame) -> None:
        scaled_data = self.scaler.transform(data)
        self.pca.partial_fit(scaled_data)
        return

    def learn(self, data: pd.DataFrame, tolerance: float = 0.01) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        pc_data = pd.DataFrame(index=data.index,
                               columns=self._pc_names,
                               data=self.pca.transform(self.scaler.transform(data)),
                               )
        for c in pc_data.columns:
            if c not in self.limits:
                self.limits[c] = {
                    'lo': pc_data[c].min(),
                    'hi': pc_data[c].max(),
                }
            mean = (pc_data[c].max() + pc_data[c].min()) / 2
            half_range = (pc_data[c].max() - pc_data[c].min()) / 2
            self.limits[c]['lo'] = min(self.limits[c]['lo'], mean - half_range * (1 + tolerance))
            self.limits[c]['hi'] = max(self.limits[c]['hi'], mean + half_range * (1 + tolerance))
        return

    def examine(self, data: pd.DataFrame) -> pd.DataFrame:
        # examine this data for anomalies
        pc_data = pd.DataFrame(index=data.index,
                               columns=self._pc_names,
                               data=self.pca.transform(self.scaler.transform(data)),
                               )
        result = pd.DataFrame(index=pc_data.index, columns=pc_data.columns, data=0, dtype='uint8')
        for c in pc_data.columns:
            if c in self.limits:
                result[c] = ~pc_data[c].between(self.limits[c]['lo'], self.limits[c]['hi'])
        return result
