
import math
import numpy as np



###########################################################
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    if not isinstance(p, (np.ndarray)):
        p=np.array(p)
    if not isinstance(q, (np.ndarray)):
        q=np.array(q)

    return np.sqrt(np.sum((p-q)**2))

###########################################################
def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*D
    y_target=math.sin(angle)*D
    return np.array([x_target,y_target])


###########################################################
def get_tgt_belief(tgt_obs,tgt_obs_uncertainty,tgt_belief,tgt_belief_uncertainty):
    # the target position estiamte across fixations (based on Kalman Filter)
    # return a belief of the target position, and its uncertainty measure

    z1,sigma1=tgt_obs,tgt_obs_uncertainty
    z2,sigma2=tgt_belief,tgt_belief_uncertainty
    # to avoid the following error
    # RuntimeWarning: invalid value encountered in double_scalars
    sigma1=max(0.0001,sigma1)
    sigma2=max(0.0001,sigma2)

    w1=sigma2**2/(sigma1**2+sigma2**2)
    w2=sigma1**2/(sigma1**2+sigma2**2)

    tgt_belief=w1*z1+w2*z2
    tgt_belief_uncertainty=np.sqrt( (sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2))

    return tgt_belief, tgt_belief_uncertainty