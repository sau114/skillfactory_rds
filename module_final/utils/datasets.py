# classes for access to various datasets

import json
import os
import os.path
import random

import pandas as pd


class DatasetError(ValueError):
    pass


class GhlKasperskyDataset:
    # GHL dataset from Kaspersky Lab

    _dt_origin = '2022-08-01'

    def __init__(self,
                 path: str = 'E:\\Datasets\\GHL',
                 ):

        # check path availability
        self.path = None
        if not os.path.isdir(path):
            raise DatasetError(f'Path {path} is not available.')
        else:
            self.path = path

        # check predefined dtypes
        self._dtypes = None
        try:
            with open(os.path.join(self.path, 'dtypes.json'), 'r') as f:
                self._dtypes = json.load(f)
        except OSError:
            raise DatasetError('File dtypes.json is not available.')
        except json.decoder.JSONDecodeError:
            raise DatasetError('File dtypes.json is not valid JSON.')

        # check required sub-directories
        subdirs = ('train',
                   'test',
                   )
        for d in subdirs:
            if not os.path.isdir(os.path.join(self.path, d)):
                raise DatasetError(f'Sub-directory {d} is not available.')

        # lists of full path to files
        self._train_files = []
        self._valid_files = []
        self._test_files = []
        # dataset-specific
        self._train_size = None
        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path})'

    def shake_not_stir(self,
                       random_state: int = 31,
                       train_size: float = 1.0,
                       ):
        random.seed(random_state)
        train_subdirs = ('train',)
        test_subdirs = ('test',)
        # list train-valid series
        filepath_list = []
        for subdir in train_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        filepath_list.append(entry.path)
        # in this dataset is only one long series for train-validation
        self._train_files = filepath_list
        self._valid_files = filepath_list
        self._train_size = train_size  # split by rows in generators
        # list testing series
        filepath_list = []
        for subdir in test_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        filepath_list.append(entry.path)
        self._test_files = random.sample(filepath_list, k=len(filepath_list))
        return

    def train_generator(self) -> tuple:
        # load train, split by rows, prepare index
        for filepath in self._train_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='time',
                               )
            data.index.name = None
            # dataset-specific
            data = data.head(round(data.shape[0] * self._train_size))
            data.index = pd.to_datetime(data.index, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def valid_generator(self) -> tuple:
        # load valid, split by rows, prepare index
        for filepath in self._valid_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='time',
                               )
            data.index.name = None
            # dataset-specific
            data = data.head(round(data.shape[0] * self._train_size))
            data.index = pd.to_datetime(data.index, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            data = data.tail(data.shape[0] - round(data.shape[0] * self._train_size))
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def test_generator(self) -> tuple:
        # load valid, split by rows, prepare index
        for filepath in self._test_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='time',
                               )
            data.index.name = None
            # dataset-specific
            data = data.head(round(data.shape[0] * self._train_size))
            data.index = pd.to_datetime(data.index, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            # choose one of the halves, anomaly is in one of them
            half = data.shape[0] // 2
            if random.randint(0, 1):
                data = data.iloc[:half]
                anomalies = anomalies.iloc[:half]
            else:
                data = data.iloc[half:]
                anomalies = anomalies.iloc[half:]
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename


class TepHarvardDataset:
    # TEP dataset from Harvard Dataverse

    _dt_origin = '2022-08-01'

    def __init__(self,
                 path: str = 'E:\\Datasets\\TEP\\dataverse',
                 ):

        # check path availability
        self.path = None
        if not os.path.isdir(path):
            raise DatasetError(f'Path {path} is not available.')
        else:
            self.path = path

        # check predefined dtypes
        self._dtypes = None
        try:
            with open(os.path.join(self.path, 'dtypes.json'), 'r') as f:
                self._dtypes = json.load(f)
        except OSError:
            raise DatasetError('File dtypes.json is not available.')
        except json.decoder.JSONDecodeError:
            raise DatasetError('File dtypes.json is not valid JSON.')

        # check required sub-directories
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
                       random_state: int = 31,
                       train_size: float = 1.0,
                       only_normal_train: bool = True,
                       balanced_test: bool = True,
                       ):
        random.seed(random_state)
        # kwarg only_normal_train
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
        filepath_list = []
        for subdir in train_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        filepath_list.append(entry.path)
        # split series to train and validation
        random.shuffle(filepath_list)
        self._train_files = filepath_list[:round(len(filepath_list) * train_size)]
        self._valid_files = filepath_list[round(len(filepath_list) * train_size):]
        # list testing series
        filepath_list = []
        for subdir in test_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        filepath_list.append(entry.path)
        # kwarg balanced_test
        if balanced_test:
            files_norm = [f for f in filepath_list if 'fault_free' in f]
            files_anom = [f for f in filepath_list if 'faulty' in f]
            n_balance = min(len(files_norm), len(files_anom))
            files_norm = random.sample(files_norm, k=n_balance)
            files_anom = random.sample(files_anom, k=n_balance)
            filepath_list = files_norm + files_anom
        self._test_files = random.sample(filepath_list, k=len(filepath_list))
        return

    def train_generator(self) -> tuple:
        # load train, prepare index
        for filepath in self._train_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='sample',
                               )
            data.index.name = None
            # dataset-specific
            # for train first 1 h (20 * 3 min) always normal
            data.loc[1:20, 'faultNumber'] = 0
            data.index = pd.to_datetime((data.index-1)*3, unit='m', origin=self._dt_origin)
            data.index.freq = '3 min'
            anomalies = data['faultNumber'].rename('anomaly')
            data.drop(columns=['faultNumber'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def valid_generator(self) -> tuple:
        # load valid, prepare index
        for filepath in self._valid_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='sample',
                               )
            data.index.name = None
            # dataset-specific
            # for train first 1 h (20 * 3 min) always normal
            data.loc[1:20, 'faultNumber'] = 0
            data.index = pd.to_datetime((data.index-1)*3, unit='m', origin=self._dt_origin)
            data.index.freq = '3 min'
            anomalies = data['faultNumber'].rename('anomaly')
            data.drop(columns=['faultNumber'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def test_generator(self) -> tuple:
        # load test, prepare index
        for filepath in self._test_files:
            data = pd.read_csv(filepath,
                               dtype=self._dtypes,
                               index_col='sample',
                               )
            data.index.name = None
            # dataset-specific
            # for train first 8 h (160 * 3 min) always normal
            data.loc[1:160, 'faultNumber'] = 0
            data.index = pd.to_datetime((data.index-1)*3, unit='m', origin=self._dt_origin)
            data.index.freq = '3 min'
            anomalies = data['faultNumber'].rename('anomaly')
            data.drop(columns=['faultNumber'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename


if __name__ == '__main__':
    # dataset = GhlKasperskyDataset()
    dataset = TepHarvardDataset()
    print(dataset)

    dataset.shake_not_stir()
    print('train files:', len(dataset._train_files))
    print('valid files:', len(dataset._valid_files))
    print('test files:', len(dataset._test_files))

    gen = dataset.train_generator()
    for series, anomalies, name in gen:
        print('train yield:', series.shape, anomalies.shape, name)

    gen = dataset.valid_generator()
    for series, anomalies, name in gen:
        print('valid yield:', series.shape, anomalies.shape, name)

    gen = dataset.test_generator()
    for series, anomalies, name in gen:
        print('test yield:', series.shape, anomalies.shape, name)
