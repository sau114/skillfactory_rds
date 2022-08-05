# classes for access to various datasets

import json
import os
import os.path

from sklearn.model_selection import train_test_split
import pandas as pd

class DatasetError(ValueError):
    pass


class TepHarvardDataset:
    # TEP dataset from Harvard Dataverse

    def __init__(self,
                 path: str = 'E:\\Datasets\\TEP\\dataverse',
                 ):

        # check path availability
        self.path = None
        if not os.path.isdir(path):
            raise DatasetError(f'Path {path} is not available.')
        else:
            self.path = path

        # check predefined dtypes_
        self._dtypes = None
        try:
            with open(os.path.join(self.path, 'dtypes.json'), 'r') as f:
                self._dtypes = json.load(f)
        except OSError:
            raise DatasetError('File dtypes.json is not available.')
        except json.decoder.JSONDecodeError:
            raise DatasetError('File dtypes.json is not valid JSON.')

        # check sub-directories
        subdirs = ('fault_free_training',
                   'fault_free_testing',
                   'faulty_training',
                   'faulty_testing',
                   )
        for d in subdirs:
            if not os.path.isdir(os.path.join(self.path, d)):
                raise DatasetError(f'Sub-directory {d} is not available.')

        # lists of full path to files
        self._train_files = []
        self._valid_files = []
        self._test_files = []

        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path})'

    def shake_not_stir(self,
                       train_size: float = 0.8,
                       random_state: int = 31,
                       only_normal_train: bool = False,
                       ):
        if only_normal_train:
            train_subdirs = ('fault_free_training',
                             )
        else:
            train_subdirs = ('fault_free_training',
                             'faulty_training',
                             )
        test_subdirs = ('fault_free_testing',
                        'faulty_testing',
                        )
        # list train-valid series
        files_list = []  # list of full paths to files
        # scan all sub-directories
        for subdir in train_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                # simple check filese
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        files_list.append(entry.path)
        # split series to train and validation
        self._train_files, self._valid_files = train_test_split(files_list,
                                                                train_size=train_size,
                                                                random_state=random_state,
                                                                shuffle=True,
                                                                )
        # list testing series
        files_list = []  # list of full paths to files
        # scan all sub-directories
        for subdir in test_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                # simple check
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        files_list.append(entry.path)
        self._test_files = files_list
        return

    def train_series_generator(self) -> tuple:
        # load train series and return series and anomaly attribute
        for filepath in self._train_files:
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].max()
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)

    def valid_series_generator(self) -> tuple:
        # load train series and return series and anomaly attribute
        for filepath in self._valid_files:
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].max()
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)

    def test_series_generator(self) -> tuple:
        # load train series and return series and anomaly attribute
        for filepath in self._test_files:
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].max()
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)


if __name__ == '__main__':
    dataset = TepHarvardDataset()
    print(dataset)

    dataset.shake_not_stir()

    gen = dataset.train_series_generator()
    for series, anomaly in gen:
        pass