def experiments(exp):
	if exp==1:
		exp_str='Fitts1954_Table1'
		Ws=np.array([2,1,0.5,0.25])
		Ds=np.array([2,4,8,16])
	elif exp==2:
		exp_str='Fitts1954 Table2 Disc Transfer'
		Ws=np.array([0.5,0.25,0.125,0.0625])
		Ds=np.array([4,8,16,32])
	elif exp==3:
		exp_str='Fitts1954 Table3 Pin Transfer'
		Ws=np.array([0.25,0.125,0.0625,0.03125])
		Ds=np.array([1,2,4,8,16])
	elif exp==4:
		exp_str='Jagacinski 1983 Helmet'
		Ws=[1.22,0.7,0.4]
		Ds=[2.45,4.28,7.5]
	elif exp==5:
		exp_str='Jagacinski 1983 Joystick'
		Ws=[0.92,0.52,0.3]
		Ds=[2.45,4.28,7.5]
	else:
		exp_str='Zhai2004'
		Ws=np.array([72,36,12])
		Ds=np.array([120,360,840])


	Ws=np.round(Ws/(np.max(Ds)*(1/0.5)),3)
	Ds=np.round(Ds/(np.max(Ds)*(1/0.5)),3)

	return Ws,Ds,exp_str

def find_MT(fitts_D,fitts_W):
	nTrials=100
	subs=np.ndarray((nTrials,1))
	mts=np.ndarray((nTrials,1))
	for trial in range(nTrials):
		dis_to_target=fitts_D
		current_pos=np.array([0,0])
		target_pos = get_new_target(fitts_D)
		aim_pos=target_pos
		sub=0
		mt=0
		while dis_to_target>fitts_W/2:
			sub+=1
			pos, vel, stage, end_pos=hand_model.motor_program(current_pos,aim_pos,
				prep_duration,stop_duration)
			dis_to_target=calc_dis(end_pos,target_pos)
			current_pos=end_pos
			mt+=len(pos)

		mts[trial]=mt
		subs[trial]=sub
	return mts,subs

def calc_rmse(mt_pred,mt_data):
    mt_pred=np.array(mt_pred)
    mt_data=np.array(mt_data)

    se=(mt_data-mt_pred)**2
    rmse=np.sqrt(np.mean(se))
    return rmse



	
if __name__ == '__main__':
	from hand_model import HandModel
	from utils import get_new_target, calc_dis
	import matplotlib.pyplot as plt
	import numpy as np

	motor_SDN=0.008
	motor_CN=0.018
	pv_constant_hand=40
	pv_slope_hand=2.5
	prep_duration,stop_duration=0,0

	hand_model=HandModel(motor_SDN,motor_CN,pv_constant_hand,pv_slope_hand)


	exp=5
	Ws,Ds,exp_str=experiments(exp)
	print(exp_str)
	print(f'Ws={Ws}')
	print(f'Ds={Ds}')

	colors=['r','g','b']
	ss=['s','o','>']
	plt.figure(figsize=(12,5))

	for iw,fitts_W in enumerate(Ws):
		for id1,fitts_D in enumerate(Ds):
			mts,subs=find_MT(fitts_D,fitts_W)


			id=np.log2((2*fitts_D)/fitts_W)
			plt.subplot(1,2,1)
			plt.plot(id,np.mean(subs),ss[iw],color=colors[id1], 
				markerfacecolor='w',markersize=10,
				label=f'W={fitts_W}')

			plt.subplot(1,2,2)
			plt.plot(id,np.mean(mts),ss[iw],color=colors[id1], 
				markerfacecolor='w',markersize=10,
				label=f'W={fitts_W}')

	plt.subplot(1,2,1)
	plt.xlim(0,6)
	plt.ylabel('num of subs')

	plt.subplot(1,2,2)
	plt.ylim(0,1400)
	plt.xlim(0,6)
	plt.ylabel('MT (ms)')
	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
	plt.tight_layout()
	
	plt.savefig('test.png')


