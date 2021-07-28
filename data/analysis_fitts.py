import numpy as np
from experiment import experiments
from eye_hand_env import EyeHandEnv

import pandas as pd
import matplotlib.pyplot as plt
import os

from eye_hand_env import EyeHandEnv
from utils import *
from core_utils import *


from local_utils import *


shapes=['^:','o:','s:','>:','*:']
colors=['#d73027',
		'#91bfdb',
		'#4575b4',
		'#fee090',
		'#fc8d59']

def experiments(exp):
    if exp==1:
        exp_str='Zhai2004_EA'
        data=pd.read_csv('data/Zhai2004_fig7.csv', decimal=',')
        width=np.array(data['width'],dtype=np.float64)
        distance=np.array(data['distance'],dtype=np.float64)
        time=np.array(data['time'],dtype=np.float64)/1000
        type1=data['type']
        data=data[type1=='EA',:]
        known_target=True
    elif exp==2:
        exp_str='Fitts1954 Table2 Disc Transfer'
        data=pd.read_csv('data/fitts54_2.csv', decimal=',')
        known_target=True
    elif exp==3:
        exp_str='Fitts1954 Table3 Pin Transfer'
        data=pd.read_csv('data/fitts54_3.csv', decimal=',')
        known_target=True
    elif exp==4:
        exp_str='Jagacinski 1983 Helmet'
        data=pd.read_csv('data/jag1983_fig4.csv', decimal=',')
        # one of the 72 targets appeared randomly next
        known_target=False
    elif exp==5:
        exp_str='Jagacinski 1983 Joystick'
        data=pd.read_csv('data/jag1983_fig5.csv', decimal=',')
        # one of the 72 targets next
        known_target=False

    else:
        print('no such data')

    return data,exp_str


data,exp_str=experiments(2)
width=np.array(data['width'],dtype=np.float64)
distance=np.array(data['distance'],dtype=np.float64)
time=np.array(data['time'],dtype=np.float64)

Ws=np.unique(width)
Ds=np.unique(distance)
Ws_transformed=np.round(Ws/(np.max(Ds)*(1/0.5)),3)
Ds_transformed=np.round(Ds/(np.max(Ds)*(1/0.5)),3)



# Model prediction
perceptual_noise=0.07
ocular_noise=0.06
ocular_CN=0.00001

motor_noise=0.02
motor_CN=0.017

train=False
n_eps=500

mt_pred=[]
mt_data=[]

for cw,fitts_W in enumerate(Ws_transformed):
	id_w=np.zeros_like(Ds) # IDs for the fitts_W
	mt_w=np.zeros_like(Ds) # MTs for the fitts_W
	for cd,fitts_D in enumerate(Ds_transformed):
		idx=np.logical_and(width==Ws[cw],distance==Ds[cd])
		mt_data.append(time[idx].item())

		# simulate each task for n_eps trials
		movement_time_all_eps=np.zeros(n_eps)

		env= EyeHandEnv(train=train,perceptual_noise=perceptual_noise,
			ocular_noise=ocular_noise,ocular_CN=ocular_CN,
			motor_noise=motor_noise,motor_CN=motor_CN,fitts_W=fitts_W,fitts_D=fitts_D)

		for eps in range(n_eps):
			obs=env.reset()
			done=False
			while not done:

				action=0
				if env.stage[HAND]==FIXATE:
					action=2
				if env.stage[EYE]==FIXATE:
					action=1
				obs, reward, done, info = env.step(action)
				if done:
					movement_time_all_eps[eps]=env.episode_steps*env.sim_time
					eps+=1

		id_w[cd]=IDs(fitts_W,fitts_D)
		mt_w[cd]=np.mean(movement_time_all_eps)/1000
		mt_pred.append(mt_w[cd])

	plt.plot(id_w,mt_w,'-',color=colors[cw], label=f'model')

	
rmse=calc_rmse(mt_pred,mt_data)




# plot data
for count_w,w in enumerate(np.unique(width)):
	plt.plot(IDs(width[width==w],distance[width==w]),time[width==w],shapes[count_w],
		color=colors[count_w],markersize=15,label=f'W={w}')
plt.xlabel(f'ID=$log2((2*D)/W)$')
plt.ylabel('Movement time (sec)')


plt.title(f'rmse={np.round(rmse,3)}')
plt.xlabel('Index of difficulty')
plt.legend()
plt.savefig(f'figs/model_{exp_str}.png')
