# classes for access to various datasets

from typing import Optional
import os
import os.path
import random
import pandas as pd


class DatasetError(ValueError):
    pass


def _scan_subdirs_for_snappy(root: str, subdirs: tuple) -> list:
    filepath_list = []
    for sd in subdirs:
        with os.scandir(os.path.join(root, sd)) as it:
            for entry in it:
                if entry.name.endswith('.snappy') and entry.is_file():
                    filepath_list.append(entry.path)
    return filepath_list


class GhlKasperskyDataset:
    # GHL dataset from Kaspersky Lab

    _dt_origin = '2022-08-01T00:00:00'

    def __init__(self,
                 path: str = 'E:\\Datasets\\GHL',
                 ):
        # check path availability
        self.path = None
        if not os.path.isdir(path):
            raise DatasetError(f'Path {path} is not available.')
        else:
            self.path = path
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
        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path})'

    def shake_not_stir(self,
                       random_state: Optional[int] = None,
                       valid_test_ratio: float = 0.0,
                       ):
        if isinstance(random_state, int):
            random.seed(random_state)
        # sub-directories with data
        train_subdirs = ('train',)
        test_subdirs = ('test',)
        # list train series
        filepath_list = _scan_subdirs_for_snappy(self.path, train_subdirs)
        self._train_files = random.sample(filepath_list, k=len(filepath_list))
        # list valid-test series
        filepath_list = _scan_subdirs_for_snappy(self.path, test_subdirs)
        random.shuffle(filepath_list)
        n_valid_files = round(len(filepath_list) * valid_test_ratio)
        self._valid_files = filepath_list[:n_valid_files]
        self._test_files = filepath_list[n_valid_files:]
        return

    def train_generator(self) -> tuple:
        # load train, prepare index
        for filepath in self._train_files:
            data = pd.read_parquet(filepath)
            data.index.name = None
            data.index = pd.to_datetime(data.index.values, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def valid_generator(self) -> tuple:
        # load valid, prepare index
        for filepath in self._valid_files:
            data = pd.read_parquet(filepath)
            data.index.name = None
            data.index = pd.to_datetime(data.index.values, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename

    def test_generator(self) -> tuple:
        # load valid, prepare index
        for filepath in self._test_files:
            data = pd.read_parquet(filepath)
            data.index.name = None
            data.index = pd.to_datetime(data.index.values, unit='m', origin=self._dt_origin)
            data.index.freq = '1 min'
            anomalies = data['attack'].rename('anomaly')
            data.drop(columns=['attack'], inplace=True)
            filename = os.path.split(filepath)[1]
            yield data, anomalies, filename


class TepHarvardDataset:
    # TEP dataset from Harvard Dataverse

    _dt_origin = '2022-08-01T00:00:00'

    def __init__(self,
                 path: str = 'E:\\Datasets\\TEP\\dataverse',
                 ):
        # check path availability
        self.path = None
        if not os.path.isdir(path):
            raise DatasetError(f'Path {path} is not available.')
        else:
            self.path = path
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
                       random_state: Optional[int] = None,
                       valid_test_ratio: float = 0.0,  # don't used
                       balanced_test: bool = False,
                       ):
        if isinstance(random_state, int):
            random.seed(random_state)
        # kwarg only_normal_train
        train_subdirs = ('fault_free_training',
                         )
        valid_subdirs = ('faulty_training',
                         )
        test_subdirs = ('fault_free_testing',
                        'faulty_testing',
                        )
        # list train-valid series
        filepath_list = _scan_subdirs_for_snappy(self.path, train_subdirs)
        self._train_files = random.sample(filepath_list, k=len(filepath_list))
        # list validation series
        filepath_list = _scan_subdirs_for_snappy(self.path, valid_subdirs)
        self._valid_files = random.sample(filepath_list, k=len(filepath_list))
        # list testing series
        filepath_list = _scan_subdirs_for_snappy(self.path, test_subdirs)
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
            data = pd.read_parquet(filepath)
            data.index.name = None
            # dataset-specific: for train first 1 h (20 * 3 min) always normal
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
            data = pd.read_parquet(filepath)
            data.index.name = None
            # dataset-specific: for valid first 1 h (20 * 3 min) always normal
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
            data = pd.read_parquet(filepath)
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
