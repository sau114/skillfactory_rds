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

    def __init__(self,
                 path: str = 'E:\\Datasets\\GHL',
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
        subdirs = ('train',
                   'test',
                   )
        for d in subdirs:
            if not os.path.isdir(os.path.join(self.path, d)):
                raise DatasetError(f'Sub-directory {d} is not available.')

        # lists of full path to files
        self._train_files = []
        self._valid_files = []
        self._train_size = None
        self._test_files = []
        # current generator filename
        self._gen_filename = None
        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path})'

    def shake_not_stir(self,
                       train_size: float = 0.8,
                       random_state: int = 31,
                       ):
        train_subdirs = ('train',
                         )
        test_subdirs = ('test',
                        )
        random.seed(random_state)
        # list train-valid series
        files_list = []  # list of full paths to files
        # scan all sub-directories
        for subdir in train_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                # simple check filese
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        files_list.append(entry.path)
        # we have only one long series for train-validation
        self._train_files = files_list
        self._valid_files = files_list
        # split this series by rows later
        self._train_size = train_size
        # list testing series
        files_list = []  # list of full paths to files
        # scan all sub-directories
        for subdir in test_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                # simple check
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        files_list.append(entry.path)
        self._test_files = random.sample(files_list, k=len(files_list))
        return

    def train_series_generator(self) -> tuple:
        # load train series and return series and anomaly attribute
        for filepath in self._train_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='time',
                                 )
            series= series.head(round(series.shape[0] * self._train_size))
            anomaly = series['attack'].rename('anomaly')
            series.drop(columns=['attack'], inplace=True)
            yield (series, anomaly)

    def valid_series_generator(self) -> tuple:
        # load valid series and return series and anomaly attribute
        for filepath in self._valid_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='time',
                                 )
            series= series.tail(series.shape[0] - round(series.shape[0] * self._train_size))
            anomaly = series['attack'].rename('anomaly')
            series.drop(columns=['attack'], inplace=True)
            yield (series, anomaly)

    def test_series_generator(self) -> tuple:
        # load test series and return series and anomaly attribute
        for filepath in self._test_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='time',
                                 )
            anomaly = series['attack'].rename('anomaly')
            series.drop(columns=['attack'], inplace=True)
            yield (series, anomaly)


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
        # current generator filename
        self._gen_filename = None
        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path})'

    def shake_not_stir(self,
                       train_size: float = 0.8,
                       random_state: int = 31,
                       only_normal_train: bool = True,
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
        random.seed(random_state)
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
        random.shuffle(files_list)
        self._train_files = files_list[:round(len(files_list)*train_size)]
        self._valid_files = files_list[round(len(files_list)*train_size):]
        # list testing series
        files_list = []  # list of full paths to files
        # scan all sub-directories
        for subdir in test_subdirs:
            with os.scandir(os.path.join(self.path, subdir)) as it:
                # simple check
                for entry in it:
                    if entry.name.endswith('.csv') and entry.is_file():
                        files_list.append(entry.path)
        self._test_files = random.sample(files_list, k=len(files_list))
        return

    def train_series_generator(self) -> tuple:
        # load train series and return series and anomaly attribute
        for filepath in self._train_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].rename('anomaly')
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)

    def valid_series_generator(self) -> tuple:
        # load valid series and return series and anomaly attribute
        for filepath in self._valid_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].rename('anomaly')
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)

    def test_series_generator(self) -> tuple:
        # load test series and return series and anomaly attribute
        for filepath in self._test_files:
            self._gen_filename = filepath
            series = pd.read_csv(filepath,
                                 dtype=self._dtypes,
                                 index_col='sample',
                                 )
            anomaly = series['faultNumber'].rename('anomaly')
            series.drop(columns=['faultNumber'], inplace=True)
            yield (series, anomaly)


if __name__ == '__main__':
    dataset = GhlKasperskyDataset()
    # dataset = TepHarvardDataset()
    print(dataset)

    dataset.shake_not_stir()
    print('train files:', len(dataset._train_files))
    print('valid files:', len(dataset._valid_files))
    print('test files:', len(dataset._test_files))

    gen = dataset.train_series_generator()
    for series, anomaly in gen:
        print('train yield:', series.shape, anomaly.shape, dataset._gen_filename)

    gen = dataset.valid_series_generator()
    for series, anomaly in gen:
        print('valid yield:', series.shape, anomaly.shape, dataset._gen_filename)

    gen = dataset.test_series_generator()
    for series, anomaly in gen:
        print('test yield:', series.shape, anomaly.shape, dataset._gen_filename)
