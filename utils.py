

import numpy as np



###########################################################
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    if not isinstance(p, (np.ndarray)):
        p=np.array(p)
    if not isinstance(q, (np.ndarray)):
        q=np.array(q)

    return np.sqrt(np.sum((p-q)**2))

