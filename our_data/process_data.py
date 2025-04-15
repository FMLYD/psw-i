from sklearn.preprocessing import StandardScaler
from pygrinder import mcar,mar_logistic,mnar_x, fill_and_get_mask_torch,fill_and_get_mask_numpy
import argparse
import numpy as np
import os
import pandas as pd
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed
from pygrinder import mcar, fill_and_get_mask_torch,fill_and_get_mask_numpy

import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers



import benchpots
import os

import h5py
import numpy as np
set_random_seed(2024)
def inverse_sliding_window(X):
    """Restore the original time-series data from the generated sliding window samples.
    Note that this is the inverse operation of the `sliding_window` function, but there is no guarantee that
    the restored data is the same as the original data considering that
    1. the sliding length may be larger than the window size and there will be gaps between restored data;
    2. if values in the samples get changed, the overlap part may not be the same as the original data after averaging;
    3. some incomplete samples at the tail may be dropped during the sliding window operation, hence the restored data
       may be shorter than the original data.

    Parameters
    ----------
    X :
        The generated time-series samples with sliding window method, shape of [n_samples, n_steps, n_features],
        where n_steps is the window size of the used sliding window method.

    sliding_len :
        The sliding length of the window for each moving step in the sliding window method used to generate X.

    Returns
    -------
    restored_data :
        The restored time-series data with shape of [total_length, n_features].

    """
    assert len(X.shape) == 3, f"X should be a 3D array, but got {X.shape}"
    n_samples, window_size, n_features = X.shape
    sliding_len =window_size
    if sliding_len >= window_size:
        if sliding_len > window_size:
            logger.warning(
                f"sliding_len {sliding_len} is larger than the window size {window_size}, "
                f"hence there will be gaps between restored data."
            )
        restored_data = X.reshape(n_samples * window_size, n_features)
    else:
        collector = [X[0][:sliding_len]]
        overlap = X[0][sliding_len:]
        for x in X[1:]:
            overlap_avg = (overlap + x[:-sliding_len]) / 2
            collector.append(overlap_avg[:sliding_len])
            overlap = np.concatenate(
                [overlap_avg[sliding_len:], x[-sliding_len:]], axis=0
            )
        collector.append(overlap)
        restored_data = np.concatenate(collector, axis=0)
    return restored_data
def read_data(data_path):
    """
    Reads a CSV file, removes the first row and first column, and returns the resulting DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The trimmed DataFrame.
    """
    # 读取CSV文件


    df = pd.read_csv(data_path)
    if 'PEMS' in data_path:
        df_trimmed = df.iloc[1:, :].reset_index(drop=True).to_numpy()
        return df_trimmed
    # 删除第一行和第一列
    if 'Pedestrian'  not in data_path:
        df_trimmed = df.iloc[1:, 1:].reset_index(drop=True).to_numpy()
        
        
        return df_trimmed
    else:
        df=df.to_numpy().astype(float).flatten()
        df=np.expand_dims(df,axis=-1)
        
        return df 
    
def process_data(data_name,mask_mode):
    def artificial_mask(mask_mode='mcar'):
        if mask_mode == 'mcar':
            return mcar
        if mask_mode=='mar':
            return mar_logistic
        if mask_mode=='mnar':
            return mnar_x
    try:
        if 'PEMS' in data_name:
            dataset=np.squeeze(np.load(f'./{data_name}.npz')['data'])
        else:
            dataset=read_data("./"+data_name+".csv")
        
    except:
        dataset=get_datasets_path(data_name)
    if mask_mode=='mcar':
        for p in [0.1,0.3,0.5,0.7]:
        
            X_observed=artificial_mask(mask_mode)(dataset,p=p)
            
            # X_observed=scaler.fit_transform(X_observed)
            # X_all=scaler.transform(dataset)
            X_all=dataset
            np.savez(f"./{data_name}_{p}_processed_data.npz", X_observed=X_observed, X_all=X_all)
    if mask_mode=='mnar':
        X_observed=np.squeeze(artificial_mask(mask_mode)(np.expand_dims(dataset,0),1))
        X_all=dataset
        np.savez(f"./{data_name}_mnar_processed_data.npz", X_observed=X_observed, X_all=X_all)
    if mask_mode=='mar':
        X_observed=artificial_mask(mask_mode)(dataset,0.5,0.1)
        X_all=dataset
        np.savez(f"./{data_name}_mar_processed_data.npz", X_observed=X_observed, X_all=X_all)

def get_datasets_path(data_dir):
    """
    We return the concatenated versions of the dataset.
    :param data_dir:
    :return:
    """
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")

    if False:
        # if LAZY_LOAD, only need to provide the dataset file path to PyPOTS models
        prepared_train_set = train_set_path
        prepared_val_set = val_set_path
    else:
        # if not LAZY_LOAD, extract and organize the data into dictionaries for PyPOTS models
        with h5py.File(train_set_path, "r") as hf:
            train_X_arr = hf["X"][:]
        with h5py.File(val_set_path, "r") as hf:
            val_X_arr = hf["X"][:]
            val_X_ori_arr = hf["X_ori"][:]

        prepared_train_set = {"X": train_X_arr}
        prepared_val_set = {"X": val_X_arr, "X_ori": val_X_ori_arr}

    with h5py.File(test_set_path, "r") as hf:
        test_X_arr = hf["X"][:]
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE

    test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    
    train_X_arr=inverse_sliding_window(train_X_arr)
    val_X_ori_arr=inverse_sliding_window(val_X_ori_arr)
    test_X_ori_arr=inverse_sliding_window(test_X_ori_arr)
    return np.concatenate([train_X_arr,val_X_ori_arr,test_X_ori_arr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset parameter.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--mask_mode', type=str, required=True, help='Path to the dataset')

    # parser.add_argument('--ratio', type=float, required=True, help='Path to the dataset')
    args = parser.parse_args()
    
    process_data(args.dataset,args.mask_mode)