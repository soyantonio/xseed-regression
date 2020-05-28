import numpy as np
import math

def mse(y : np.ndarray, hyp : np.ndarray) -> np.ndarray:
    """
    MSE Cost function
    Uses Quadratic loss, L2 Loss
    """
    m = y.size
    helper = y - hyp
    return np.dot(helper, helper)/(m)

def mae(y : np.ndarray, hyp : np.ndarray) -> np.ndarray:
    """
    MAE Cost function
    Uses L1 Loss
    """
    m = y.size
    return abs(y - hyp)/(m)


def rmse(y : np.ndarray, hyp : np.ndarray) -> np.ndarray:
    """
    RMSE Cost function
    Root of Quadratic loss
    """
    m = y.size
    helper = y - hyp
    cost = np.dot(helper, helper)/(m)
    return math.sqrt(cost)