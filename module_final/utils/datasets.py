# classes for access to various datasets

from typing import Optional
import os
import os.path
import random
import pandas as pd


class DatasetError(ValueError):
    pass


class BatchIterator:

    def __init__(self, batch_files: list):
        self.batch_files = batch_files
        self.index = 0
        pass

    def __len__(self):
        return len(self.batch_files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.batch_files):
            raise StopIteration
        filename = self.batch_files[self.index]
        batch = pd.read_parquet(filename)
        # parquet forgot frequency
        batch.index.freq = f'{(batch.index[1] - batch.index[0]).seconds} s'
        # split to data and anomalies
        anomalies = batch['anomaly']
        data = batch.drop(columns=['anomaly'])
        info = os.path.split(filename)[1]
        self.index += 1
        return data, anomalies, info


class Dataset:
    # root class for Datasets

    ROOT = 'a:\\datasets\\this_dataset'

    def _check_required_subdirs(self):
        subdirs = ('',)
        for sd in subdirs:
            if not os.path.isdir(os.path.join(self.ROOT, sd)):
                raise DatasetError(f'Sub-directory {sd} is not available.')
        return

    def __init__(self, path: Optional[str] = None):
        # common
        if path is not None:
            self.ROOT = path
        if not os.path.isdir(self.ROOT):
            raise DatasetError(f'Path {self.ROOT} is not available.')
        # specific
        self._check_required_subdirs()
        # common
        self.train_files = []
        self.valid_files = []
        self.test_files = []
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.ROOT})'

    def _list_subdirs_for(self, extension: str, subdirs: Optional[tuple]) -> list:
        files_list = []
        if subdirs is None:
            subdirs = ('',)  # scan root
        for sd in subdirs:
            with os.scandir(os.path.join(self.ROOT, sd)) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(extension):
                        files_list.append(entry.path)
        return files_list

    def _list_files_package(self, kind: str) -> list:
        files_list = []
        if kind == 'train':
            pass
        elif kind == 'valid':
            pass
        elif kind == 'test':
            pass
        else:
            raise DatasetError(f'Unsupported kind of files: {kind}')
        random.shuffle(files_list)
        return files_list

    def shake_not_stir(self,
                       random_state: Optional[int] = None,
                       valid2test_ratio: float = 0.3,
                       ):
        # common
        random.seed(random_state)
        # specific
        train_files = self._list_files_package('train')
        valid_files = self._list_files_package('valid')
        test_files = self._list_files_package('test')
        # common
        self.train_files = train_files
        n_valid = min(round(len(test_files) * valid2test_ratio), len(valid_files))
        self.valid_files = random.sample(valid_files, k=n_valid)
        self.test_files = test_files
        pass

    def train_generator(self):
        return BatchIterator(self.train_files)

    def valid_generator(self):
        return BatchIterator(self.valid_files)

    def test_generator(self):
        return BatchIterator(self.test_files)


class GhlKasperskyDataset(Dataset):
    # GHL dataset from Kaspersky Lab

    ROOT = 'E:\\Datasets\\GHL'

    def _check_required_subdirs(self):
        subdirs = ('train',
                   'test',
                   )
        for sd in subdirs:
            if not os.path.isdir(os.path.join(self.ROOT, sd)):
                raise DatasetError(f'Sub-directory {sd} is not available.')
        return

    def _list_files_package(self, kind: str) -> list:
        if kind == 'train':
            files_list = self._list_subdirs_for('.snappy', subdirs=('train',))
        elif kind == 'valid':
            files_list = self._list_subdirs_for('.snappy', subdirs=('test',))
        elif kind == 'test':
            files_list = self._list_subdirs_for('.snappy', subdirs=('test',))
        else:
            raise DatasetError(f'Unsupported kind of files: {kind}')
        random.shuffle(files_list)
        return files_list


class TepHarvardDataset(Dataset):
    # TEP dataset from Harvard Dataverse

    ROOT = 'E:\\Datasets\\TEP\\dataverse'

    def _check_required_subdirs(self):
        subdirs = ('fault_free_training',
                   'fault_free_testing',
                   'faulty_training',
                   'faulty_testing',
                   )
        for sd in subdirs:
            if not os.path.isdir(os.path.join(self.ROOT, sd)):
                raise DatasetError(f'Sub-directory {sd} is not available.')
        return

    def _list_files_package(self, kind: str) -> list:
        if kind == 'train':
            files_list = self._list_subdirs_for('.snappy', subdirs=('fault_free_training',))
        elif kind == 'valid':
            files_list = self._list_subdirs_for('.snappy', subdirs=('faulty_training',))
        elif kind == 'test':
            files_list = self._list_subdirs_for('.snappy', subdirs=('fault_free_testing', 'faulty_testing'))
        else:
            raise DatasetError(f'Unsupported kind of files: {kind}')
        random.shuffle(files_list)
        return files_list


class TepKasperskyDataset(Dataset):
    # TEP dataset from Kaspersky Lab

    ROOT = 'E:\\Datasets\\TEP\\kaspersky'

    def _check_required_subdirs(self):
        subdirs = ('_pretreated',
                   )
        for sd in subdirs:
            if not os.path.isdir(os.path.join(self.ROOT, sd)):
                raise DatasetError(f'Sub-directory {sd} is not available.')
        return

    def _list_files_package(self, kind: str) -> list:
        files_list = self._list_subdirs_for('.snappy', subdirs=('_pretreated',))
        single_state_list = [f for f in files_list if 'single_state_mode' in f]  # 200
        transient_list = [f for f in files_list if 'transient_mode' in f]  # 346
        attack_list = [f for f in files_list if 'attack_mode' in f]  # 142
        if kind == 'train':
            # equal from single_state and transient
            n_transient = min(len(transient_list), len(single_state_list))
            files_list = single_state_list + random.sample(transient_list, k=n_transient)
        elif kind == 'valid':
            # equal from transient and attack
            n_valid = min(len(transient_list), len(attack_list))
            files_list = random.sample(transient_list, k=n_valid) + random.sample(attack_list, k=n_valid)
        elif kind == 'test':
            # all from transient and attack
            files_list = transient_list + attack_list
        else:
            raise DatasetError(f'Unsupported kind of files: {kind}')
        random.shuffle(files_list)
        return files_list


class SwatItrustDataset(Dataset):
    # SWaT dataset from iTrust

    ROOT = 'E:\\Datasets\\SWaT\\datasetA1'

    def _check_required_subdirs(self):
        subdirs = ('',
                   )
        for sd in subdirs:
            if not os.path.isdir(os.path.join(self.ROOT, sd)):
                raise DatasetError(f'Sub-directory {sd} is not available.')
        return

    def _list_files_package(self, kind: str) -> list:
        files_list = self._list_subdirs_for('.snappy', subdirs=('',))
        if kind == 'train':
            files_list = [f for f in files_list if 'SWaT_Dataset_Normal' in f]
        elif kind == 'valid':
            files_list = [f for f in files_list if 'SWaT_Dataset_Attack' in f]
        elif kind == 'test':
            files_list = [f for f in files_list if 'SWaT_Dataset_Attack' in f]
        else:
            raise DatasetError(f'Unsupported kind of files: {kind}')
        random.shuffle(files_list)
        return files_list
