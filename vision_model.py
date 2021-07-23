'''
Intrinsic position uncertainty is modelled as 
an intrinsic position noise whose standard
deviation varies as a linear function of eccentricity.

reference:
Michel, M., & Geisler, W. S. (2011). 
Intrinsic position uncertainty explains detection and localization performance 
in peripheral vision. 
Journal of Vision, 11(1), 18-18.
'''

import numpy as np
import math
from utils import calc_dis
###########################################################


class VisionModel():
	def __init__(self, perceptual_noise):
		self.perceptual_noise=perceptual_noise

	def get_tgt_obs(self,fixate_pos,target_pos):
		x1,y1=fixate_pos[0],fixate_pos[1]
		x2,y2=target_pos[0],target_pos[1]
		angle=math.atan((y1-y2)/(x1-x2))

		ecc=calc_dis(fixate_pos,target_pos)
		sigma=self.perceptual_noise*ecc

		# diagonal covariance
		cov = [[sigma, 0], [0, 3*sigma/4]]  

		x, y = np.random.multivariate_normal([0,0], cov)

		xnew=x*math.cos(angle)-y*math.sin(angle)+x2
		ynew=x*math.sin(angle)+y*math.cos(angle)+y2

		tgt_obs=np.array([xnew,ynew])

		return tgt_obs,sigma






if __name__=='__main__':
	import matplotlib.pyplot as plt
	perceptual_noise=0.09
	vm=VisionModel(perceptual_noise)



	fmt=['ro','bs','c*','m+','yd','k.']
	scale_deg=60
	fixate=np.array([0,0])
	fig=plt.figure(figsize=(12,5))

	targets=np.array([[0,0.5*scale_deg],
		[0.4*scale_deg,0],
		[0,-0.3*scale_deg],
		[-0.2*scale_deg,0]])

	N=2000
	for n,target in enumerate(targets):
		ecc=calc_dis(fixate,target)
		std=np.round(ecc*perceptual_noise,2)
		
		obs=np.zeros((N,2))
		for i in range(N):

			obs[i,:],U=vm.get_tgt_obs(fixate,target)
		
		ax = fig.add_subplot(122)
		if n==0:
			plt.plot(obs[:,0],obs[:,1],'ko',markerfacecolor='w',label='Indicated location')		
			plt.plot(fixate[0],fixate[1],'gs',markersize=12,label='Fixation')
		else:
			plt.plot(obs[:,0],obs[:,1],'ko',markerfacecolor='w')		
			plt.plot(fixate[0],fixate[1],'gs',markersize=12)
		plt.plot(target[0],target[1],fmt[n],markersize=12,linewidth=2,
			label=f'Tgt ecc={ecc}')

			
	plt.xlabel('degree')
	plt.ylabel('degree')
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.title(f'Perceptual noise={perceptual_noise}')
	ax.set_aspect('equal', adjustable='box')

	ax2 = fig.add_subplot(121)
	tmp=obs-target
	tmp=tmp/ecc
	plt.plot(tmp[:,0],tmp[:,1],'ko',markerfacecolor='w',label='Indicated location')
	plt.plot(0,0,'r+',markersize=12,label='Target')
	plt.xlabel('Normalized radial error (error/eccentricity)')
	plt.ylabel('Normalized tangential error (error/eccentricity)')	
	plt.xlim(-0.55,0.55)
	plt.ylim(-0.55,0.55)
	ax2.set_aspect('equal', adjustable='box')
	plt.xticks(np.arange(-0.5,0.55,0.25))
	plt.yticks(np.arange(-0.5,0.55,0.25))
	plt.legend()
	plt.savefig('figures/VisionModelTest.png')
