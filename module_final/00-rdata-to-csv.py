#!/usr/bin/env python
# coding: utf-8

# In[1]:


# only once
# !pip install pyreadr


# IMPORTANT: One of dataset is too large for loading in jupyter. You can get "MemoryError: Unable to allocate 3.79 GiB for an array with shape (53, 9600000) and data type float64".

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
DTYPES_FILE = 'dtypes.json'  # dtypes of columns


# In[5]:


proc = psutil.Process(os.getpid())

def print_memusage(prefix=''):
# print memory usage info
    print(prefix, f'{proc.memory_info().rss/1024**2:0.2f} MB')


# # Convert RData

# In[6]:


def clear_tags(df: pd.DataFrame) -> None:
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
    }
    
    df.rename(columns=rename_dict, inplace=True)
    return


# In[7]:


def optimize_dtypes(df: pd.DataFrame) -> None:
    # optimize dataframe by memory usage
    uint_columns = df.columns.values[:3]  # this columns can be uint
    float_columns = df.columns.values[3:]  # other must be float
    
    df[uint_columns] = df[uint_columns].apply(pd.to_numeric, downcast='unsigned')
    df[float_columns] = df[float_columns].apply(pd.to_numeric, downcast='float')
    
    # saving our dtypes description for further use
    dtypes_file = os.path.join(DIR, DTYPES_FILE)
    if not os.path.isfile(dtypes_file):
        # we need to create it
        names = df.dtypes.index  # columns names
        types = [c.name for c in df.dtypes]  # columns types
        dtypes_dict = dict(zip(names, types))  # dict for pandas.read_csv
        with open(dtypes_file, 'w') as f:
            json.dump(dtypes_dict, f)
        
    return


# In[8]:


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


# In[9]:


print_memusage('Before loading')
print()

for f in R_FILES:
    r_file = os.path.join(DIR, f)
    r_data = pyreadr.read_r(r_file)
    print_memusage('After reading ' + f)

    for k in r_data.keys():
        print('Dataset', k, 'with shape', r_data[k].shape)
        clear_tags(r_data[k])
        optimize_dtypes(r_data[k])
        print_memusage('After optimizing')
        
        split_n_save(r_data[k], k)
        print_memusage("After split\'n\'saving")
        
    del r_data  # because need a lot of RAM
    print_memusage('After deleting')
    print()

