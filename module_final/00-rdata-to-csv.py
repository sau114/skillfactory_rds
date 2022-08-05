#!/usr/bin/env python
# coding: utf-8

# In[1]:


# only once
# !pip install pyreadr


# IMPORTANT: One of dataset is too large for loading in jupyter.
# You can get "MemoryError: Unable to allocate 3.79 GiB for an array with shape (53, 9600000) and data type float64".

# # Prepare

# In[2]:


# import libraries
import json
import os
import os.path
import pandas as pd
import pyreadr


# In[3]:


# check memory usage
import psutil


# In[4]:


DIR = 'E:\\Datasets\\TEP\\dataverse'  # dataset source dir
R_FILES = ('TEP_FaultFree_Training.RData',  # datasets RData, sorted by size
           'TEP_FaultFree_Testing.RData',
           'TEP_Faulty_Training.RData',
           'TEP_Faulty_Testing.RData',
           )
DTYPES_FILE = 'dtypes.json'  # dtypes_ of columns


# In[5]:


proc = psutil.Process(os.getpid())

def print_memusage(prefix=''):
# print memory usage info
    print(prefix, f'{proc.memory_info().rss/1024**2:0.2f} MB')


# # Convert RData

# In[6]:


def optimize_dtypes(df: pd.DataFrame) -> None:
    # optimize dataframe by memory usage
    uint_columns = df.columns.values[:3]  # this columns can be uint
    float_columns = df.columns.values[3:]  # other must be float
    
    df[uint_columns] = df[uint_columns].apply(pd.to_numeric, downcast='unsigned')
    df[float_columns] = df[float_columns].apply(pd.to_numeric, downcast='float')
    
    # saving our dtypes_ description for further use
    dtypes_file = os.path.join(DIR, DTYPES_FILE)
    if not os.path.isfile(dtypes_file):
        # we need to create it
        names = df.dtypes.index  # columns names
        types = [c.name for c in df.dtypes]  # columns types
        dtypes_dict = dict(zip(names, types))  # dict for pandas.read_csv
        with open(dtypes_file, 'w') as f:
            json.dump(dtypes_dict, f)
        
    return


# In[7]:


def split_n_save(df: pd.DataFrame, subdir: str) -> None:
    outdir = os.path.join(DIR, subdir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    for flt in df['faultNumber'].unique():
        for run in df['simulationRun'].unique():
            sub_df = (df.query(f'faultNumber == {flt} and simulationRun == {run}')
                        .drop(columns=['simulationRun'])
                        .set_index('sample')
                     )
            fname = os.path.join(outdir, f'{subdir}_run_{run:03d}_fault_{flt:03d}.csv')
            sub_df.to_csv(fname)

    return


# In[8]:


print_memusage('Before loading')
print()

for f in R_FILES:
    r_file = os.path.join(DIR, f)
    r_data = pyreadr.read_r(r_file)
    print_memusage('After reading ' + f)

    for k in r_data.keys():
        print('Dataset', k, 'with shape', r_data[k].shape)
        optimize_dtypes(r_data[k])
        print_memusage('After optimizing')
        
        split_n_save(r_data[k], k)
        print_memusage("After split'n'saving")
        
    del r_data  # because need a lot of RAM
    print_memusage('After deleting')
    print()

