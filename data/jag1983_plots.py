import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def IDs(W,D):
	W=np.array(W)
	D=np.array(D)
	return np.log2((D*2)/W)

Ds=[2.45,4.28,7.5] # visual angle in degree

shapes=['^:','o:','s:','>:','*:']

colors=['#d73027',
'#91bfdb',
'#4575b4','#fee090','#fc8d59',]


plt.figure(figsize=(12,6))
for condi in [0,1]:
	if condi==0:
		data=pd.read_csv('data/jag1983_fig4.csv', decimal=',')
		str1='helmet-mounted'
	else:
		data=pd.read_csv('data/jag1983_fig5.csv', decimal=',')
		str1='joystick'


	width=np.array(data['width'],dtype=np.float64)
	distance=np.array(data['distance'],dtype=np.float64)
	time=np.array(data['time'],dtype=np.float64)


	# plot data
	
	for count_w,w in enumerate(np.unique(width)):
		plt.subplot(1,2,condi+1)
		plt.plot(IDs(width[width==w],distance[width==w]),time[width==w],shapes[count_w],
			color=colors[count_w],markersize=15,label=f'W={w}')
		plt.xlabel(f'ID=$log2((2*D)/W)$')
		plt.ylabel('Movement time (sec)')


	xx=IDs(width,distance)
	z=np.polyfit(xx,time, 1)

	xxx=np.linspace(np.min(xx)-0.5,np.max(xx)+0.5,100)
	yyy=xxx*z[0]+z[1]
	plt.plot(xxx,yyy,color='k',label='Fitts Law')
	plt.title(str1)
	plt.xlim(0,6.5)
	plt.ylim(0,1.45)

	plt.legend(title='D=[2.45,4.28,7.5] degrees')
plt.savefig(f'figs/jag1983.png')
