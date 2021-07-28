import numpy as np

def ID(W,D):
	return np.log2(2*D/W)

def ID_shannon(W,D):
	return np.log2((D+W)/W)

def inches_to_cm(inches):
	to_cm=2.54
	return inches*to_cm

def pixel_to_cm(num_pixel):
	#Each pixel on the screen was 0.02055 cm wide
	each_pixel_cm=0.02055
	return each_pixel_cm*num_pixel

def experiments(exp):
	if exp==1:
		exp_str='Fitts1954_Pen'
		Ws=np.array([2,1,0.5,0.25])
		Ds=np.array([2,4,8,16])
	elif exp==2:
		exp_str='Fitts1954_Disc'
		Ws=np.array([0.5,0.25,0.125,0.0625])
		Ds=np.array([4,8,16,32])
	elif exp==3:
		exp_str='Fitts1954_Pin'
		Ws=np.array([0.25,0.125,0.0625,0.03125])
		Ds=np.array([1,2,4,8,16])
	elif exp==4:
		exp_str='Jagacinski1983_Helmet'
		Ws=[1.22,0.7,0.4]
		Ds=[2.45,4.28,7.5]
	elif exp==5:
		exp_str='Jagacinski1983_Joystick'
		Ws=[0.92,0.52,0.3]
		Ds=[2.45,4.28,7.5]
	else:
		exp_str='Zhai2004'
		Ws=np.array([72,36,12])
		Ds=np.array([120,360,840])


	Ws=np.round(Ws/(np.max(Ds)*(1/0.5)),3)
	Ds=np.round(Ds/(np.max(Ds)*(1/0.5)),3)

	return Ws,Ds,exp_str

if __name__ == '__main__':
	import itertools
	for exp in [2,3,4,5,6]:
		Ws,Ds,exp_str=experiments(exp)
		print(exp_str)
		print(f'Ws={Ws}')
		print(f'Ds={Ds}')
