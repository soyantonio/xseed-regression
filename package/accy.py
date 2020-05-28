import numpy as np

def r2(y : np.ndarray, fs : np.ndarray):
    ymean = np.mean(y)
    helper1 = y - ymean
    ss_tot = np.dot(helper1, helper1)
    helper2 = y - fs
    ss_res = np.dot(helper2, helper2)
    r2 = 1 - ss_res/ss_tot
    return r2