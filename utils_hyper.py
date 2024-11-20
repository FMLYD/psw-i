import torch, random, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hyperimpute.plugins.utils.simulate import simulate_nan
import pandas as pd

def enable_reproducible_results(seed: int = 0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


def scale_data(X):
    X = np.asarray(X)
    preproc = MinMaxScaler()
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def simulate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7]):
    X = scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            datasets[ampute_mechanism][p_miss] = ampute(X, ampute_mechanism, p_miss)

    return datasets


# third party
import numpy as np


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = np.absolute(X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]])
                _output = _output.sum() / mask_[:, i].sum()
                output += [str(_output.round(5))]

        return output


def RMSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        RMSE : np.ndarray

    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = (X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]]) ** 2
                _output = _output.sum() / mask_[:, i].sum()
                _output = np.sqrt(_output)
                output += [str(_output.round(5))]

        return output


__all__ = ["MAE", "RMSE"]
