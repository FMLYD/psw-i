import math
import argparse
import logging
import os
import sys
import time
from typing import Union
import ot
import numpy as np
import torch
import yaml
import torch.nn as nn
from model_others import KNNImputation, MissForestImputation, MVAImputation
from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed
from pygrinder import mcar, fill_and_get_mask_torch,fill_and_get_mask_numpy
from utils import get_log_path
from global_config import (
    TORCH_N_THREADS,
    RANDOM_SEEDS,
)
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers

import benchpots
from hpo_results import HPO_RESULTS
import os

import h5py
import numpy as np
from sklearn.base import TransformerMixin
from global_config import LAZY_LOAD_DATA

DEVICE = "cuda:1"

def get_log_path(model, dataset):
    argv = sys.argv[1:]
    argv = [arg for arg in argv if 'gpu_id' not in arg and 'wandb_project' not in arg and 'dataset_fold_path' not in arg
            and "saving_path" not in arg and "device" not in arg and "impute_all_datasets" not in arg
            ]
    logfilepath = "{}/{}-{}-{}".format(model, model, dataset, '_'.join(argv))
    logfilepath="./output/"+logfilepath
    return logfilepath


def inverse_sliding_window(X):
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

def sliding_window(
    time_series: Union[np.ndarray, torch.Tensor],
    window_len: int,
    sliding_len: int = None,
) -> Union[np.ndarray, torch.Tensor]:
    sliding_len = window_len if sliding_len is None else sliding_len
    total_len = time_series.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len

    # remove the last one if left length is not enough
    if total_len - start_indices[-1] < window_len:
        left_len = total_len - start_indices[-1]
        to_drop = math.floor(window_len / sliding_len)
        logger.warning(
            f"{total_len}-{start_indices[-1]}={left_len} < {window_len}. "
            f"The last {to_drop} samples are dropped due to the left length {left_len} is not enough."
        )
        start_indices = start_indices[:-to_drop]

    sample_collector = []
    for idx in start_indices:
        sample_collector.append(time_series[idx : idx + window_len])

    if isinstance(time_series, torch.Tensor):
        samples = torch.cat(sample_collector, dim=0)
    elif isinstance(time_series, np.ndarray):
        samples = np.asarray(sample_collector).astype("float32")
    else:
        raise RuntimeError

    return samples
def artificial_mask(X,p=0.1,mask_mode='mcar'):
    if mask_mode == 'mcar':
        X=mcar(X,p)
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="the dataset name",
        default="ETTh1"
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        default="data/generated_datasets/italy_air_quality_rate01_step12_point "
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        help="the saving path of the model and logs",
        default="results_point_rate01"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run the model, e.g. cuda:0",
        default="cuda"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        help="the number of rounds running the model to get the final results ",
        default=1,
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=".",
        required=False,
    )
    parser.add_argument(
        "--impute_all_sets",
        help="whether to impute all sets or only the test set",
        action="store_true",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002
        )
    parser.add_argument(
            "--n_epochs",
            type=int,
            default=100
        )
    parser.add_argument(
            "--n_pairs",
            type=int,
            default=2
        )
    parser.add_argument(
            "--reg_sk",
            type=float,
            default=0.005
        )
    parser.add_argument(
            "--batch_size",
            type=int,
            default=64
        )
    parser.add_argument(
            "--normalize",
            type=int,
            default=1
        )
    parser.add_argument(
            "--seq_length",
            type=int,
            default=24
        )
    parser.add_argument(
            "--mva_kernel",
            type=int,
            default=24
        )
    parser.add_argument(
            "--distance",
            type=str,
            default="fft"
        )
    parser.add_argument(
            "--ot_type",
            type=str,
            default="uot_mm"
        )
    parser.add_argument(
            "--reg_m",
            type=float,
            default="1",
            help="the strength of KL divergence specified in UOT"
        )
    parser.add_argument(
            "--dropout",
            type=float,
            default="0",
            help="the strength of KL divergence specified in UOT"
        )
    args = parser.parse_args()
    #set the logger
    """
    Set the outpath of the model, we will select some hyperparameters of the experiment as the path of the folder, 
    since the args now is a little  complicated.
    """

    args_cmd = "_".join(sys.argv)
    # path = f"./results_finetune_sinkhorn_knn/{args_cmd}"
    path=f"./results/{args.dataset}/{args.ratio}/PSW-I"
    initializer = None
    # initializer = KNNImputation(k=50, weights="distance")
    initializer = MVAImputation(window_length=args.mva_kernel, window_type='exponential')
    DEVICE=args.device
    class OTImputationIni(TransformerMixin):
        """Sinkhorn imputation can be used to impute quantitative data and it relies on the idea that two batches extracted randomly from the same dataset should share the same distribution and consists in minimizing optimal transport distances between batches.

    Args:
        eps: float, default=0.01
            Sinkhorn regularization parameter.
        lr : float, default = 0.01
            Learning rate.
        opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
            Optimizer class to use for fitting.
        n_epochs : int, default=15
            Number of gradient updates for each model within a cycle.
        batch_size : int, defatul=256
            Size of the batches on which the sinkhorn divergence is evaluated.
        n_pairs : int, default=10
            Number of batch pairs used per gradient update.
        noise : float, default = 0.1
            Noise used for the missing values initialization.
        scaling: float, default=0.9
            Scaling parameter in Sinkhorn iterations
        """

        def __init__(
            self,
            lr: float = 1e-2,
            opt = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-2,
            reg_sk: float = 1,
            numItermax: int = 1000,
            stopThr = 1e-9,
            normalize = 0,
            initializer = None,
            replace = False,
            seq_length = 16,
            distance = 1,
            ot_type = 'sinkhorn',
            reg_m = 10,dropout=0.1
        ):
            self.lr = lr
            self.opt = opt
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.n_pairs = n_pairs
            self.noise = noise
            self.reg_sk = reg_sk
            self.numItermax = numItermax
            self.stopThr = stopThr
            self.normalize = normalize
            self.initializer = initializer
            self.replace = replace
            self.seq_length = seq_length
            self.distance = distance
            self.ot_type = ot_type
            self.dropout=nn.Dropout(p=dropout)
            self.reg_m = reg_m

        def fit_transform(self, X: pd.DataFrame, X_all, *args, **kwargs) -> pd.DataFrame:
            all_mask=np.isnan(X)
            val=X
            train=mcar(X,p=0.2)
            mask = np.isnan(train)
            val_mask=np.isnan(train) ^(np.isnan(X_all))
            maes=[]
            mses=[]
            mres=[]
            val_maes=[]
            val_mses=[]
            val_mres=[]
            tick=0
            if self.initializer is not None:
                imps = self.initializer.fit_transform(train)
                
                while np.any(np.isnan(imps)):
                    temp_mask=np.isnan(imps)
                    imps[temp_mask] = self.initializer.fit_transform(imps)[temp_mask]
                imps=imps
                imps = torch.tensor(imps).double().to(DEVICE)
                imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
            X = torch.tensor(X).to(DEVICE)
            X = X.clone()
            train = torch.tensor(train).to(DEVICE)
            train = train.clone()
            val = torch.tensor(val).to(DEVICE)
            val = val.clone()
            n, d = X.shape

            if self.batch_size > n // 2:
                e = int(np.log2(n // 2))
                self.batch_size = 2**e

            if self.initializer is None:
                imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]


            imps = imps.to(DEVICE)
            mask = torch.tensor(mask).to(DEVICE)
            all_mask = torch.tensor(all_mask).to(DEVICE)
            val_mask = torch.tensor(val_mask).to(DEVICE)

            imps.requires_grad = True

            optimizer = self.opt([imps], lr=self.lr)

            for i in range(self.n_epochs):
                X_filled = train.detach().clone().double()
                X_filled[mask.bool()] = imps.double()
                loss = 0
                tick+=1
                for _ in range(self.n_pairs):
                    optimizer.zero_grad()
                    coeff = 1
                    idx1 = np.random.choice((n - self.seq_length)//coeff, self.batch_size, replace=self.replace) * coeff
                    idx2 = np.random.choice((n - self.seq_length)//coeff, self.batch_size, replace=self.replace) * coeff

                    X1 = torch.stack([(X_filled[idx:idx+self.seq_length]) for idx in idx1], dim=0)
                    X2 = torch.stack([X_filled[idx:idx+self.seq_length] for idx in idx2], dim=0)
                    if self.distance == 'time':
                        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
                    elif self.distance == 'fft':
                        X1 = torch.fft.rfft(X1.transpose(1, 2))
                        X2 = torch.fft.rfft(X2.transpose(1, 2))
                        M = (self.dropout(((X1.flatten(1)[:,None,:]) - X2.flatten(1)[None,:,:]).abs())).sum(-1)
                    elif self.distance == 'fft_mag':
                        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
                        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
                        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
                    elif self.distance == 'fft_mag_abs':
                        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
                        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
                        M = torch.norm(X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:], p=1, dim=2)
                    if self.normalize == 1:
                        M = M / M.max()
                    a, b = torch.ones((self.batch_size,), device=M.device) / self.batch_size, torch.ones((self.batch_size,), device=M.device) / self.batch_size
                    if self.ot_type == 'sinkhorn':
                        pi = ot.sinkhorn(a, b, M, reg=self.reg_sk, max_iter=self.numItermax, tol_rel=self.stopThr).detach()
                    elif self.ot_type == 'emd':
                        pi = ot.emd2(a, b, M, numItermax=self.numItermax).detach()
                    elif self.ot_type == 'uot':
                        pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=self.reg_sk, stopThr=self.stopThr, numItermax=self.numItermax, reg_m=self.reg_m).detach()
                    elif self.ot_type == 'uot_mm':
                        pi = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=self.reg_m, c=None, reg=0, div='kl', G0=None, numItermax=self.numItermax, stopThr=self.stopThr).detach()
                    loss = loss + (pi * M).sum() / self.n_pairs

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        # Catch numerical errors/overflows (should not happen)
                        
                        break

                    
                loss.backward()
                optimizer.step()
                print(loss.item())
                X_filled = train.detach().clone().double()
                X_filled[mask.bool()] = imps
                X_imputed = X_filled.detach().cpu().numpy()

                mae = calc_mae(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())
                mse = calc_mse(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())
                mre = calc_mre(X_imputed, np.nan_to_num(X_all), all_mask.detach().cpu().numpy())
                
                maes.append(mae)
                mses.append(mse)
                mres.append(mre)
                val_mae = calc_mae(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
                val_mse = calc_mse(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
                val_mre = calc_mre(X_imputed, np.nan_to_num(X_all), val_mask.detach().cpu().numpy())
                print("tick:",tick,"val_MAE:", round(val_mae,5), "val_MSE:", round(val_mse,5),"test_MAE:", round(mae,5), "test_MSE:", round(mse,5))
                if i>0 and (val_mae<min(val_maes) )  :
                    tick=0

                val_maes.append(val_mae)
                val_mses.append(val_mse)
                val_mres.append(val_mre)
                if tick>10:
                    break
                    

            temp=[]
            for i in range(len(val_maes)):
                temp.append(val_maes[i]/max(val_maes)+val_mses[i]/max(val_mses))
            
            loc=temp.index(min(temp))
            return X_filled.detach().cpu().numpy(), maes[loc],mses[loc],mres[loc]




    model = OTImputationIni(batch_size=args.batch_size, lr=args.lr, n_epochs=args.n_epochs, n_pairs=args.n_pairs, reg_sk=args.reg_sk, noise=1e-4, numItermax=1000, stopThr=1e-6, initializer=initializer, normalize=args.normalize, seq_length=args.seq_length, distance=args.distance, ot_type=args.ot_type, reg_m=args.reg_m,dropout=args.dropout)
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

    file_handler = logging.FileHandler(f"{path}/log.txt")
    logger.addHandler(file_handler)

    # set the number of threads for pytorch

    """    
        build benchmark
    """
    torch.set_num_threads(TORCH_N_THREADS)

    dataset_file_name=f"{args.dataset}_{args.ratio}_PSW-I_data.npz"
    data =  np.load(f"./our_data/{args.dataset}_{args.ratio}_processed_data.npz")
    X_observed=data['X_observed']
    X_all=data['X_all']

    mae_collector = []
    mse_collector = []
    mre_collector = []
    time_collector = []
    """" process model for parametric models and mask again for the validation """


    result_saving_path = os.path.join(args.saving_path, f"sinkhorn_{args.dataset}")
    for n_round in range(args.n_rounds):
        set_random_seed(RANDOM_SEEDS[n_round])
        round_saving_path = os.path.join(result_saving_path, f"round_{n_round}")

        # get the hyperparameters and setup the model



        """
        if the model is parametric, we will train it with our processed dataset, otherwise we will use the X.
        """
        
        # X=pd.DataFrame(X_observed)
        X_imputation,mae,mse,mre = model.fit_transform(X_observed.copy(), X_all)
        
        X_imputation=np.array(X_imputation)
        if not os.path.exists(f"./imputated_datasets/{args.dataset}/"):
            os.makedirs(f"./imputated_datasets/{args.dataset}/")
        start_time = time.time()
        time_collector.append(time.time() - start_time)

        indicating_mask = np.isnan(X_all) ^ np.isnan(X_observed)
        # mae = calc_mae(X_imputation, np.nan_to_num(X_all), indicating_mask)
        # mse = calc_mse(X_imputation, np.nan_to_num(X_all), indicating_mask)
        # mre = calc_mre(X_imputation, np.nan_to_num(X_all), indicating_mask)
        mae_collector.append(mae)
        mse_collector.append(mse)
        mre_collector.append(mre)

        # if impute_all_sets is True, impute the train and val sets for saving
        



        pickle_dump(
            {
                "train_set_imputation": X_imputation,
                "val_set_imputation": X_imputation,
                "test_set_imputation": X_imputation,
            },
            os.path.join(round_saving_path, "imputation.pkl"),
        )
        logger.info(
            f"Round{n_round} - sinkhorn on {args.dataset}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
        )

    mean_mae, mean_mse, mean_mre = (
        np.mean(mae_collector),
        np.mean(mse_collector),
        np.mean(mre_collector),
    )
    std_mae, std_mse, std_mre = (
        np.std(mae_collector),
        np.std(mse_collector),
        np.std(mre_collector),
    )
    print(mae_collector)
    # num_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(
        f"Done! Final results:\n"
        # f"Averaged sinkhorn ({num_params:,} params) on {args.dataset}: "
        f"MAE={mean_mae:.4f} , "
        f"MSE={mean_mse:.4f} , "
        f"MRE={mean_mre:.4f} , "
        f"average inference train_model.py={np.mean(time_collector):.2f}"
    )
    with open(path+"/config.yaml", "w") as f:
        """
        """
        # yaml.dump()
        pass
    with open(path+"/performance.yaml",'w') as file:
        result={'MAE':float(f'{mean_mae:.4f}'), "MSE":float(f'{mean_mse:.4f}'),"MRE":float(f"{mean_mre:.4f}"),'STD_MAE':float(f"{std_mae:.4f}"),'STD_MRE':float(f"{std_mre:.4f}"),'STD_MSE':float(f"{std_mse:.4f}")}
        yaml.dump(result,file)
