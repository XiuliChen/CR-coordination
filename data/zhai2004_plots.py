import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ID_shannon(W,D):
	return np.log2((D+W)/W)

def IDs(W,D):
	W=np.array(W)
	D=np.array(D)
	return np.log2((D*2)/W)

shapes=['^-','o-','s-','>-','*-']
colors=['#d73027',
'#91bfdb',
'#4575b4','#fee090','#fc8d59',]
plt.figure(figsize=(10,10))
for condi in [0,1,2,3,4]:
	condis=['EF','F','N','A','EA']
	data=pd.read_csv('data/Zhai2004_fig7.csv', decimal=',')


	width=np.array(data['width'],dtype=np.float64)
	distance=np.array(data['distance'],dtype=np.float64)
	time=np.array(data['time'],dtype=np.float64)/1000
	type1=data['type']
	cc=condis[condi]

	width=width[type1==cc]
	distance=distance[type1==cc]
	Ds=np.unique(distance)
	time=time[type1==cc]

	# plot data
	
	for count_w,w in enumerate(np.unique(width)):
		plt.plot(ID_shannon(width[width==w],distance[width==w]),time[width==w],shapes[count_w],color=colors[condi],markersize=10,label=f'{condis[condi]}, W={w}')


	plt.legend(loc='center left', bbox_to_anchor=(1, 0.75),title=f'D={Ds} pixels')
	plt.tight_layout()
	plt.xlabel(f'ID=$log2((D+W)/W)$')
	plt.ylabel('Movement time (sec)')
	plt.ylim(0,1.650)


plt.savefig(f'figs/Zhai2004_fig7.png')




