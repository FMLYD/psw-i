# stdlib
from typing import Any, List

# third party
from geomloss import SamplesLoss
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import torch
import ot
from utils_hyper import enable_reproducible_results

# hyperimpute absolute
from hyperimpute.plugins.core.device import DEVICE
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators


class OTImputation(TransformerMixin):
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
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-2,
        reg_sk: float = 1,
        numItermax: int = 1000,
        stopThr = 1e-9,
        normalize = 0,
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

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True
        
        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), device=M.device) / self.batch_size, torch.ones((self.batch_size,), device=M.device) / self.batch_size
                
                loss = loss + ot.sinkhorn2(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class OTPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-2,
        random_state: int = 0,
        reg_sk: float = 1,
        numItermax = 1000,
        stopThr = 1e-9,
        normalize = 1,
        
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

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

        self._model = OTImputation(
            lr=lr,
            opt=opt,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            noise=noise,
            reg_sk=reg_sk,
            numItermax=numItermax,
            stopThr=stopThr,
            normalize=self.normalize
        )

    @staticmethod
    def name() -> str:
        return "ot"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("lr", [1e-2]),
            params.Integer("n_epochs", 100, 500, 100),
            params.Categorical("batch_size", [512]), 
            params.Categorical("n_pairs", [1]), 
            params.Categorical("noise", [1e-3, 1e-4]),
            params.Categorical("reg_sk", [0.1, 1, 5]),
            params.Categorical("numItermax", [1000]),
            params.Categorical("stopThr", [1e-3, 1e-9]),

        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "OTPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X)


class OTLapImputation(TransformerMixin):
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
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-4,
        numItermax: int = 1000,
        stopThr = 1e-9,
        numItermaxInner: int = 1000,
        stopThrInner = 1e-9,
        normalize = 0,
        reg_sim = 'knn',
        reg_simparam = 5,
        reg_eta = 1,
        opt_moment1 = 0.9,
        opt_moment2 = 0.999,
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta
        self.opt_moment1 = opt_moment1
        self.opt_moment2 = opt_moment2

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr, betas=(self.opt_moment1, self.opt_moment2))

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                # print(X1.dtype, X2.dtype)
                M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size
                gamma = ot.da.emd_laplace(a, b, X1, X2, M, 
                        sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta, alpha=0.5, 
                        numItermax=self.numItermax, stopThr=self.stopThr, numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                loss = loss + (gamma * M).sum()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class OTLapPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-4,
        random_state: int = 0,
        numItermax = 1000,
        stopThr = 1e-9,
        numItermaxInner: int = 1000,
        stopThrInner = 1e-9,
        normalize = 1,
        reg_sim = 'knn',
        reg_simparam = 5,
        reg_eta = 1,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self._model = OTLapImputation(
            lr=lr,
            opt=opt,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            noise=noise,
            numItermax=numItermax,
            stopThr=stopThr,
            stopThrInner=self.stopThrInner,
            numItermaxInner=self.numItermaxInner,
            normalize=self.normalize,
            reg_sim=self.reg_sim,
            reg_simparam=self.reg_simparam,
            reg_eta=self.reg_eta,
        )

    @staticmethod
    def name() -> str:
        return "ot-l"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("lr", [1e-2]),
            params.Integer("n_epochs", 100, 500, 100),
            params.Categorical("batch_size", [512]), 
            params.Categorical("n_pairs", [1]), 
            params.Categorical("noise", [1e-3, 1e-4]),
            params.Categorical("numItermax", [1000]),
            params.Categorical("stopThr", [1e-3]),
            params.Categorical("numItermaxInner", [1000]),
            params.Categorical("stopThrInner", [1e-3]),
            params.Categorical("reg_eta", [1e-2, 1e-1, 5e-1, 1, 5, 1e1]),
            params.Categorical("reg_sim", ["knn", "gauss"]),
            params.Categorical("reg_simparam", [3, 5, 7, 9]),

        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "OTPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X)


class OTRRimputation(TransformerMixin):
    """
    Round-Robin imputer with a batch sinkhorn loss
    """
    def __init__(self,
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 n_epochs=100,
                 niter=2, 
                 batch_size=512,
                 n_pairs=10, 
                 noise=1e-4,
                 numItermax = 1000,
                 stopThr = 1e-3,
                 normalize = 0,
                 reg_sk = 1,
                 weight_decay=1e-5, 
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):

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


        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize
        self.reg_sk = reg_sk

        self.niter = niter
        self.weight_decay=weight_decay
        self.order=order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol=tol
        for i in range(d): ## predict the ith variable using d-1 others
            self.models[i] = torch.nn.Linear(d - 1, 1, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))    
        order_ = torch.argsort(mask.sum(0))
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = [self.opt(self.models[i].parameters(),lr=self.lr, weight_decay=self.weight_decay) for i in range(d)]

        
        X_filled = X.clone()
        X_filled[mask.bool()] = imps

        for i in range(self.n_epochs):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            for j in order_:
                # j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()

                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()

                    for _ in range(self.n_pairs):
                        
                        idx1 = np.random.choice(n, self.batch_size, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batch_size, replace= self.batch_size > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batch_size, replace=False)
                            X2 = X_filled[idx2]
                        M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                        if self.normalize == 1:
                            M = M / M.max()
                        a, b = torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size
                        gamma = ot.sinkhorn(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                        loss = loss + (gamma * M).sum()

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()
                # print(i, j, k)
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled.detach().cpu().numpy()


class OTLapRRimputation(TransformerMixin):
    """
    Round-Robin imputer with a batch sinkhorn loss

    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"

    """
    def __init__(self,
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 n_epochs=100,
                 niter=2, 
                 batch_size=512,
                 n_pairs=10, 
                 noise=1e-4,
                 numItermax = 1000,
                 stopThr = 1e-3,
                 numItermaxInner: int = 1000,
                 stopThrInner = 1e-3,
                 normalize = 1,
                 reg_sim = 'knn',
                 reg_simparam = 5,
                 reg_eta = 1,
                 weight_decay=1e-5, 
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):
        
        
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self.niter = niter
        self.weight_decay=weight_decay
        self.order=order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol=tol
        for i in range(d): ## predict the ith variable using d-1 others
            self.models[i] = torch.nn.Linear(d - 1, 1, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))    
        order_ = torch.argsort(mask.sum(0))
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = [self.opt(self.models[i].parameters(),lr=self.lr, weight_decay=self.weight_decay) for i in range(d)]

        
        X_filled = X.clone()
        X_filled[mask.bool()] = imps

        for i in range(self.n_epochs):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            for j in order_:
                # j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()

                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()

                    for _ in range(self.n_pairs):
                        
                        idx1 = np.random.choice(n, self.batch_size, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batch_size, replace= self.batch_size > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batch_size, replace=False)
                            X2 = X_filled[idx2]
                        M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                        if self.normalize == 1:
                            M = M / M.max()
                        a, b = torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size
                        gamma = ot.da.emd_laplace(a, b, X1, X2, M, 
                                  sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta, alpha=0.5, 
                                  numItermax=self.numItermax, stopThr=self.stopThr, numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                        loss = loss + (gamma * M).sum()

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()
                # print(i, j, k)
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled.detach().cpu().numpy()

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously 
        fitted on other data.
        
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """


        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X[mask] = np.nanmean(X)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):

                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
        return X_filled

class OTLapMTLimputation():

    def __init__(self,
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 n_epochs=100,
                 niter=15, 
                 batch_size=512,
                 n_pairs=10, 
                 noise=0.1,
                 numItermax = 1000,
                 stopThr = 1e-9,
                 numItermaxInner: int = 1000,
                 stopThrInner = 1e-9,
                 normalize = 1,
                 reg_sim = 'knn',
                 reg_simparam = 5,
                 reg_eta = 1,
                 weight_decay=1e-5, 
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):
        
        
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self.niter = niter
        self.weight_decay=weight_decay
        self.order=order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol=tol
        self.models = torch.nn.Linear(d, d, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        # X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))    
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = self.opt(self.models.parameters(),lr=self.lr, weight_decay=self.weight_decay)

        
        X_filled = X.clone()
        X_filled[mask.bool()] = imps
        torch.autograd.set_detect_anomaly(True)
        for i in range(self.n_epochs):

            # if self.order == 'random':
            #     order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            # for j in order_:
            #     # j = order_[l].item()
            #     n_not_miss = (~mask[:, j].bool()).sum().item()

            #     if n - n_not_miss == 0:
            #         continue  # no missing value on that coordinate

            for k in range(self.niter):

                loss = 0

                X_filled_ = X_filled.detach()
                impute = self.models(X_filled.detach())
                X_filled_ = torch.where(mask.bool(), impute, X_filled)
                # X_filled_[mask.bool()] = impute[mask.bool()].squeeze()
                

                for _ in range(self.n_pairs):
                    
                    # idx1 = np.random.choice(n, self.batch_size, replace=False)
                    # X1 = X_filled[idx1]

                    # if self.unsymmetrize:
                    #     n_miss = (~mask[:, j].bool()).sum().item()
                    #     idx2 = np.random.choice(n_miss, self.batch_size, replace= self.batch_size > n_miss)
                    #     X2 = X_filled[~mask[:, j].bool(), :][idx2]

                    # else:
                    #     idx2 = np.random.choice(n, self.batch_size, replace=False)
                    #     X2 = X_filled[idx2]
                    idx1 = np.random.choice(n, self.batch_size, replace=False)
                    idx2 = np.random.choice(n, self.batch_size, replace=False)

                    X1 = X_filled_[idx1]
                    X2 = X_filled_[idx2]
                    M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                    if self.normalize == 1:
                        M = M / M.max()
                    a, b = torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size
                    gamma = ot.da.emd_laplace(a, b, X1, X2, M, 
                                sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta, alpha=0.5, 
                                numItermax=self.numItermax, stopThr=self.stopThr, numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                    loss = loss + (gamma * M).sum()

                optimizers.zero_grad()
                loss.backward()
                optimizers.step()

                # Impute with last parameters
                with torch.no_grad():
                    impute = self.models(X_filled.detach())
                    X_filled_ = torch.where(mask.bool(), impute, X_filled)

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously 
        fitted on other data.
        
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """


        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X[mask] = np.nanmean(X)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):

                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
        return X_filled
    

class TDMImputation(TransformerMixin):
    
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-2,
        reg_sk: float = 1,
        numItermax: int = 1000,
        stopThr = 1e-9,
        normalize = 1,
        net_depth = 1,
        net_indim = 8,
        net_hidden = 32,
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

        import FrEIA.framework as Ff
        import FrEIA.modules as Fm
        def subnet_fc(dims_in, dims_out):
            return torch.nn.Sequential(torch.nn.Linear(dims_in, net_hidden, dtype=torch.double), torch.nn.SELU(),  torch.nn.Linear(net_hidden, net_hidden, dtype=torch.double), torch.nn.SELU(),
                                torch.nn.Linear(net_hidden,  dims_out, dtype=torch.double)).to(DEVICE)
        self.projector = Ff.SequenceINN(*(net_indim,))
        for _ in range(net_depth):
            self.projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)


    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                X1_p, _ = self.projector(X1)
                X2_p, _ = self.projector(X2)
                M = ot.dist(X1_p, X2_p, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double, device=M.device) / self.batch_size
                # gamma = ot.sinkhorn(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                gamma = ot.emd(a, b, M, numItermax=self.numItermax)
                loss = loss + (gamma * M).sum()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()