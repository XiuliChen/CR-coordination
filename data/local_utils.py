
import numpy as np
import pandas as pd

def IDs(W,D,shannon=False):
    W=np.array(W)
    D=np.array(D)
    if shannon:
        return np.log2((D+W)/W)
    else:
        return np.log2(2*D/W)

def calc_rmse(mt_pred,mt_data):
    mt_pred=np.array(mt_pred)
    mt_data=np.array(mt_data)

    se=(mt_data-mt_pred)**2
    rmse=np.sqrt(np.mean(se))
    return rmse



