import pandas as pd


class WatchmanError(ValueError):
    pass


class SimpleWatchman:
    # Store limits of values.
    # Consider values outside of limits to be anomalies.

    def __init__(self):
        self.limits = {}  # dict of watched values
        return

    def learn(self, data: pd.DataFrame, tolerance: float = 0.05) -> None:
        # learn and store limits of this data
        # watchman don't forget previous limits
        for c in data.columns:
            if c not in self.limits:
                self.limits[c] = {
                    'lo': data[c].min(),
                    'hi': data[c].max(),
                }
            mean = (data[c].max() + data[c].min()) / 2
            half_span = (data[c].max() - data[c].min()) / 2
            self.limits[c]['lo'] = min(self.limits[c]['lo'], mean - half_span * (1 + tolerance))
            self.limits[c]['hi'] = max(self.limits[c]['hi'], mean + half_span * (1 + tolerance))
        return

    def examine(self, data: pd.DataFrame) -> pd.Series:
        # examine this data for anomalies
        result = pd.Series(index=data.index, data=0, dtype='uint8')
        for c in data.columns:
            if c in self.limits:
                result += data[c] < self.limits[c]['lo']
                result += data[c] > self.limits[c]['hi']
        return result
