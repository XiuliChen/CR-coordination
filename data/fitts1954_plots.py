import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import os
'''
from envs.utils import *
from envs.my_constants import *
from experiment import experiments
from envs.EyeHandEnv import EyeHandEnv
'''
###########################################################
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    return np.sqrt(np.sum((p-q)**2))


def IDs(W,D):
	W=np.array(W)
	D=np.array(D)
	return np.log2((D*2)/W)


shapes=['^:','o:','s:','>:','*:']

colors=['#d73027',
'#91bfdb',
'#4575b4','#fee090','#fc8d59',]


plt.figure(figsize=(12,6))
for condi in [0,1]:
	if condi==0:
		data=pd.read_csv('data/fitts54_2.csv', decimal=',')

		str1='disc transfer'
	else:
		data=pd.read_csv('data/fitts54_3.csv', decimal=',')
		str1='pin transfer'


	width=np.array(data['width'],dtype=np.float64)
	distance=np.array(data['distance'],dtype=np.float64)
	Ds=np.unique(distance)

	time=np.array(data['time'],dtype=np.float64)


	# plot data
	
	for count_w,w in enumerate(np.unique(width)):
		plt.subplot(1,2,condi+1)
		plt.plot(IDs(width[width==w],distance[width==w]),time[width==w],shapes[count_w],
			color=colors[count_w],markersize=10,label=f'W={w}')

		plt.xlabel(f'ID=$log2((2*D)/W)$')
		plt.ylabel('Movement time (sec)')

	'''
	xx=IDs(width,distance)
	z=np.polyfit(xx,time, 1)

	xxx=np.linspace(np.min(xx)-0.5,np.max(xx)+0.5,100)
	yyy=xxx*z[0]+z[1]
	plt.plot(xxx,yyy,color='k',label='Fitts Law')
	'''
	plt.title(str1)
	plt.xlim(2,11)
	plt.ylim(0.2,1.2)


	plt.legend(title=f'D={Ds} inches')
plt.savefig(f'figs/fitts1954.png')


