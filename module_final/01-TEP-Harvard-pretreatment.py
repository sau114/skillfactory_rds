#!/usr/bin/env python
# coding: utf-8

# IMPORTANT: One of dataset is too large for loading in jupyter. You can get "MemoryError: Unable to allocate 3.79 GiB for an array with shape (53, 9600000) and data type float64".

# # Prepare

# In[1]:


# import libraries
import os
import os.path
import pandas as pd
import pyreadr
from tqdm import tqdm


# In[2]:


SOURCE = 'Y:\\ZBU\\_Datasets\\TEP\\dataverse'  # dataset source dir
R_FILES = ('TEP_FaultFree_Training.RData',  # datasets RData, sorted by size
           'TEP_FaultFree_Testing.RData',
           'TEP_Faulty_Training.RData',
           'TEP_Faulty_Testing.RData',
        )
TARGET = 'E:\\Datasets\\TEP\\dataverse'


# # Convert RData

# In[3]:


def rename_columns(df: pd.DataFrame) -> None:
    # rename columns
    rename_dict = {
        "xmeas_1": "inp_a_flow_ksm3h",
        "xmeas_2": "inp_d_flow_kgh",
        "xmeas_3": "inp_e_flow_kgh",
        "xmeas_4": "inp_c_flow_ksm3h",
        "xmeas_5": "recyl_flow_ksm3h",
        "xmeas_6": "react_flow_ksm3h",
        "xmeas_7": "react_press_kpa",
        "xmeas_8": "react_level_pc",
        "xmeas_9": "react_temp_gc",
        "xmeas_10": "purge_flow_ksm3h",
        "xmeas_11": "seprt_temp_gc",
        "xmeas_12": "seprt_level_pc",
        "xmeas_13": "seprt_press_kpa",
        "xmeas_14": "seprt_flow_m3h",
        "xmeas_15": "strip_level_pc",
        "xmeas_16": "strip_press_kpa",
        "xmeas_17": "prod_flow_m3h",
        "xmeas_18": "strip_temp_gc",
        "xmeas_19": "steam_flow_kgh",
        "xmeas_20": "compr_power_kw",
        "xmeas_21": "re_cl_temp_gc",
        "xmeas_22": "co_cl_temp_gc",
        "xmeas_23": "react_a_prt_molp",
        "xmeas_24": "react_b_prt_molp",
        "xmeas_25": "react_c_prt_molp",
        "xmeas_26": "react_d_prt_molp",
        "xmeas_27": "react_e_prt_molp",
        "xmeas_28": "react_f_prt_molp",
        "xmeas_29": "purge_a_prt_molp",
        "xmeas_30": "purge_b_prt_molp",
        "xmeas_31": "purge_c_prt_molp",
        "xmeas_32": "purge_d_prt_molp",
        "xmeas_33": "purge_e_prt_molp",
        "xmeas_34": "purge_f_prt_molp",
        "xmeas_35": "purge_g_prt_molp",
        "xmeas_36": "purge_h_prt_molp",
        "xmeas_37": "prod_d_prt_molp",
        "xmeas_38": "prod_e_prt_molp",
        "xmeas_39": "prod_f_prt_molp",
        "xmeas_40": "prod_g_prt_molp",
        "xmeas_41": "prod_h_prt_molp",
        "xmv_1": "inp_d_feed_pc",
        "xmv_2": "inp_e_feed_pc",
        "xmv_3": "inp_a_feed_pc",
        "xmv_4": "inp_c_feed_pc",
        "xmv_5": "compr_valv_pc",
        "xmv_6": "purge_feed_pc",
        "xmv_7": "seprt_feed_pc",
        "xmv_8": "strip_feed_pc",
        "xmv_9": "steam_feed_pc",
        "xmv_10": "re_cl_feed_pc",
        "xmv_11": "co_cl_feed_pc",
        'sample': 'time',
        'faultNumber': 'anomaly',
        'simulationRun': 'run'
    }
    df.rename(columns=rename_dict, inplace=True)
    return


# In[4]:


def index_and_downsample(df: pd.DataFrame) -> None:
    # set index
    df['time'] = pd.to_datetime((df['time'].astype('int')-1)*3, unit='m', origin='2017-07-06T00:00:00')
    df.set_index('time', inplace=True)
    df.index.name = None
    # downsample don't need
    # df = df.resample('1 min').first()
    return df


# In[5]:


def optimize_dtypes(df: pd.DataFrame) -> None:
    # optimize dataframe by memory usage
    uint_columns = [
        'anomaly',
        'run',
    ]
    float_columns = [c for c in df.columns if c not in uint_columns]
    
    df[uint_columns] = df[uint_columns].apply(pd.to_numeric, downcast='unsigned')
    df[float_columns] = df[float_columns].apply(pd.to_numeric, downcast='float')
    
    return


# In[6]:


def split_n_save(df: pd.DataFrame, subdir: str) -> None:
    outdir = os.path.join(TARGET, subdir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for anomaly in tqdm(df['anomaly'].unique()):
        for run in df['run'].unique():
            sub_df = df.query(f'anomaly=={anomaly} and run=={run}').drop(columns=['run'])
            # mark normal period
            if 'training' in subdir:
                # first 1 hour
                sub_df.loc[:'2017-07-06T01:00:00', 'anomaly'] = 0
            elif 'testing' in subdir:
                # first 8 hours
                sub_df.loc[:'2017-07-06T08:00:00', 'anomaly'] = 0
            fname = os.path.join(outdir, f'{subdir}_run_{run:03d}_fault_{anomaly:03d}.snappy')
            sub_df.to_parquet(fname, compression='snappy')
    return


# In[7]:


if not os.path.isdir(TARGET):
    os.mkdir(os.path.join(TARGET))

for f in R_FILES:
    r_file = os.path.join(SOURCE, f)
    r_data = pyreadr.read_r(r_file)
    for k in r_data.keys():
        print('Dataset', k, 'with shape', r_data[k].shape)
        rename_columns(r_data[k])
        index_and_downsample(r_data[k])
        optimize_dtypes(r_data[k])
        split_n_save(r_data[k], k)


# In[11]:


r_data['fault_free_training'].index


# # Self-Check

# In[8]:


data = pd.read_parquet(os.path.join(TARGET, 'fault_free_training', 'fault_free_training_run_001_fault_000.snappy'))
data.info()


# In[9]:


data.index


# In[ ]:




