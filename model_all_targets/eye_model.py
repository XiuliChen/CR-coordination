'''
OcularModel generates a motor program given the current position and 
aim position in x,y coordinates

 Reference:
    W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
    'A parametric model for saccadic eye movement.'
    IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
    DOI: 10.1109/SPMB.2016.7846860.

'''
import math
import numpy as np
from constants import *
from utils import calc_dis
from utils_plots import plot_program

class EyeModel():
    def __init__(self,motor_SDN, motor_CN):
        self.motor_SDN=motor_SDN
        self.motor_CN=motor_CN

    def motor_program(self,current_pos,aim_pos,prep_duration,stop_duration):
        stage=[PREP]*prep_duration
        pos=[current_pos]*prep_duration
        vel=[0]*prep_duration

        # if the planned moving distance is smaller than the minimal saccade ampliude
        # check paper: Accuracy and precision of small saccades
        # 14'-20' (arcmin)
        # An arcminute (denoted by the symbol ‘), is an angular measurement equal to 1/60 of a degree or 60 arcseconds.')

        # Moving
        # add motor noise at the end point
        aim_dis=calc_dis(current_pos,aim_pos)
        _sigma= self.motor_SDN*aim_dis+self.motor_CN   
        end_pos=np.random.normal(aim_pos, _sigma,aim_pos.shape)
        amp=calc_dis(current_pos, end_pos)*SCALE_DEG

        trajectory,velocity,tmp=vel_profiles(amp)
        
        if len(trajectory)>0:
            argmax_velocity=np.argmax(velocity)
            for i in range(len(trajectory)):
                r=trajectory[i]/amp
                pos.append(current_pos+r*(end_pos-current_pos))
                vel.append(velocity[i])  
                if i<argmax_velocity:
                    stage.append(RAMPUP)
                else:
                    stage.append(RAMPDOWN)

        # fixating
        pos.extend([end_pos]*stop_duration)
        vel.extend([0]*stop_duration)  
        stage.extend([STILL]*stop_duration)

        return pos, vel, stage, end_pos


    

#############################################################################

def vel_profiles(amplitude):
    # Time axis
    Fs = 1000                           # sampling rate (samples/sec)
    t = np.arange(-0.1, 0.1+1.0/Fs, 1.0/Fs) # time axis (sec)
    # (no units)
    threshold=20 # the velocity threshold (deg/s), below this is considered as 'stop moving'.
    trajectory, velocity, pv = vel_model(t, amplitude) 
    idx=np.where(velocity<threshold)

    trajectory=np.delete(trajectory,idx)
    velocity=np.delete(velocity,idx)-threshold
    t1=np.delete(t,idx)
    stage=np.where(t1<0,0.5,1) 

    return trajectory,velocity,pv


def vel_model(t, amplitude,eta=480.0, c=8,  t0=0.0, s0=0.0):
    """
    ### Xiuli copied from https://codeocean.com/capsule/8467067/tree/v1
    ### 14 Oct 2020
    A parametric model for saccadic eye movement.
    This function simulates saccade waveforms using a parametric model.
    The saccade model corresponds to the 'main sequence' formula:
        Vp = eta*(1 - exp(-A/c))
    where Vp is the peak saccadic velocity and A is the saccadic amplitude.
    
    Usage:
        waveform, velocity, peak_velocity = 
            saccade_model(t, [eta,] [c,] [amplitude,] [t0,] [s0])
    
    Input:
        t         : time axis (sec)
        eta       : main sequence parameter (deg/sec)
        c         : main sequence parameter (no units)
        amplitude : amplitude of saccade (deg)
        t0        : saccade onset time (sec)
        s0        : initial saccade angle (degree)

    Output:
        waveform      : time series of saccadic angle
        velocity      : time series of saccadic angular velocity
        peak_velocity : peak velocity of saccade

    Reference:
    W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
    'A parametric model for saccadic eye movement.'
    IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
    DOI: 10.1109/SPMB.2016.7846860.
    """
    
    fun_f = lambda t: t*(t>=0)+0.25*np.exp(-2*t)*(t>=0)+0.25*np.exp(2*t)*(t<0)
    fun_df = lambda t: 1*(t>=0)-0.5*np.exp(-2*t)*(t>=0)+0.5*np.exp(2*t)*(t<0)
    tau = amplitude/eta         # tau: amplitude parameter (amplitude = eta*tau)
    
    if t0 == 0:
        t0 = -tau/2             # saccade onset time (sec)
    
    waveform = c*fun_f(eta*(t-t0)/c) - c*fun_f(eta*(t-t0-tau)/c) + s0
    velocity = eta*fun_df(eta*(t-t0)/c) - eta*fun_df(eta*(t-t0-tau)/c)
    peak_velocity = eta * (1 - np.exp(-amplitude/c))
    
    return waveform, velocity, peak_velocity





if __name__=="__main__":
    import matplotlib.pyplot as plt
    print('THIS IS UNIT TESTING FOR THE EYE MODEL')

    em=EyeModel(motor_SDN=0.01,motor_CN=0.001)

    current_pos=np.array([0,0])
    aim_pos=np.array([0.5,0.1])
    prep_duration=150
    stop_duration=50

    pos,vel,stage,end_pos=em.motor_program(current_pos,aim_pos,prep_duration,stop_duration)

    plot_program(current_pos,aim_pos,end_pos,pos,vel,SCALE_DEG,stage)
    plt.suptitle('Given star_pos and aim_pos, the eye model generates the trajectory between start and aim+noise')
    plt.savefig('figures/EyeModelTest.png')



    '''
    print('Figure 2: the speed profile used in the model')
    print('We only look at the visual angle udner 30, as that for bigger visual angle the eye-head-hand coordiantion becomes more proper')


    plt.figure(figsize=(6,6))
    amps=np.array([5,10,15,20,25,30])



    
    times=[]
    pv=[]

    for amplitude in amps:
        traj, traj_vel, peak_velocity =vel_profiles(amplitude)
        pv.append(np.round(peak_velocity,2))
        times.append(len(traj))

        plt.plot(traj_vel,label=f'amp={amplitude}')


    plt.xlabel("Time [ms]")
    plt.ylabel("Velocity [deg/s]")
    plt.ylim(0,500)
    plt.xticks(np.arange(-50,320,50))
    plt.yticks(np.arange(0,510,100))
    plt.title('the speed profile used in the model')
    plt.legend()
    plt.grid()





    plt.show()
    '''



