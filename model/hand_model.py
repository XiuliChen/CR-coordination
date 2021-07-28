'''
The parameters are based on:
Reference: Vercher, J. L., Magenes, G., Prablanc, C., & Gauthier, G. M. (1994). 
Eye-head-hand coordination in pointing at visual targets: 
spatial and temporal analysis. 
Experimental brain research, 99(3), 507-523.

# The hand model corresponds to the 'main sequence' formula:
peak_velocity (degree/second) = 40 + 2.5 x amplitude
duration (ms)=240 + 4.5 x amplitude,
'''


import numpy as np
from constants import *
from utils import calc_dis
from utils_plots import plot_program

class HandModel():
    def __init__(self,motor_SDN,motor_CN,pv_constant_hand,pv_slope_hand):
        self.motor_SDN=motor_SDN
        self.motor_CN=motor_CN
        self.pv_constant_hand=pv_constant_hand
        self.pv_slope_hand=pv_slope_hand

    def motor_program(self,current_pos,aim_pos,prep_duration,stop_duration):
        # Preping
        stage=[PREP]*prep_duration
        pos=[current_pos]*prep_duration
        vel=[0]*prep_duration

        # Moving
        aim_dis=calc_dis(current_pos,aim_pos)

        # add motor noise at the end point
        end_pos=np.random.normal(aim_pos,self.motor_SDN*aim_dis+self.motor_CN,aim_pos.shape)
        amp=calc_dis(current_pos, end_pos)*SCALE_DEG
       
        trajectory,velocity=self.hand_trajectory(amp,current_pos,end_pos)
        
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


        # hand stop
        pos.extend([end_pos]*stop_duration)
        vel.extend([0]*stop_duration)  
        stage.extend([STILL]*stop_duration)


        return pos, vel, stage, end_pos

    def hand_trajectory(self,amp,current_pos,actual_pos):
        '''
        peak_velocity=a + b*amp 
        different a,b for different devices.
        ''' 
        peak_velocity=self.pv_constant_hand + self.pv_slope_hand*amp

        # the minimum jerk for trajectory
        average_velocity = peak_velocity/1.875
        current = 0.0
        setpoint = amp
        frequency = 1000
        time = (setpoint - current) / average_velocity
        
        # calculate the moving distance
        # mjtg function in utils.py
        trajectory,velocity=mjtg(current, setpoint, frequency, time)

        return trajectory,velocity


###########################################################

###########################################################
# used in the hand_model to generate trajectory
def mjtg(current, setpoint, frequency, move_time):
    trajectory = []
    trajectory_derivative = []
    timefreq = int(move_time * frequency)

    for time in range(1, timefreq):
        trajectory.append(
            current + (setpoint - current) *
            (10.0 * (time/timefreq)**3
             - 15.0 * (time/timefreq)**4
             + 6.0 * (time/timefreq)**5))

        trajectory_derivative.append(
            frequency * (1.0/timefreq) * (setpoint - current) *
            (30.0 * (time/timefreq)**2.0
             - 60.0 * (time/timefreq)**3.0
             + 30.0 * (time/timefreq)**4.0))

    return trajectory, trajectory_derivative


if __name__=='__main__':
    from utils_plots import *

    motor_SDN=0.1
    motor_CN=0.001
    pv_constant_hand=40
    pv_slope_hand=2.5


    hm=HandModel(motor_SDN=motor_SDN,motor_CN=motor_CN,
        pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand)

    current_pos=np.array([0,0])
    aim_pos=np.array([0.3,0.1])
    prep_duration=150
    stop_duration=50
    

    pos,vel,stage,end_pos=hm.motor_program(current_pos,aim_pos,prep_duration,stop_duration)

    plot_program(current_pos,aim_pos,end_pos,pos,vel,SCALE_DEG,stage)
    plt.suptitle('Given star_pos and aim_pos, the hand model generates the trajectory between start and aim+noise')
    plt.savefig('figures/HandModelTest.png')

    '''
    # Set up and calculate trajectory.
    amps=np.array([5,10,20,30,40])


    times=[]
    pv=[]
    plt.figure(figsize=(7,7))
    for amplitude in amps:
        peak_velocity=40 + 2.5*amplitude
        pv.append(np.round(peak_velocity,2))
        average_velocity = peak_velocity/1.87
        current = 0.0
        setpoint = amplitude
        frequency = 1000
        time = (setpoint - current) / average_velocity
        times.append(np.round(time*1000))
        traj, traj_vel = mjtg(current, setpoint, frequency, time)

        # Create plot.
        xaxis = [i / frequency for i in range(1, int(time * frequency))]
        plt.subplot(2,1,1)
        plt.plot(traj,label=f'amp={amplitude}')
        plt.subplot(2,1,2)
        plt.plot(traj_vel,label=f'amp={amplitude}')


    plt.subplot(2,1,1)
    plt.xlabel("Time [ms]")
    plt.ylabel("Move angle [deg]")
    plt.xticks(times)
    plt.yticks(amps)
    plt.grid()

    plt.subplot(2,1,2)
    plt.xlabel("Time [ms]")
    plt.ylabel("Velocity [deg/s]")
    plt.xticks(times)
    plt.yticks(pv)
    plt.grid()

    plt.suptitle('From  Vercher et al., 2011')
    plt.legend()
    plt.show()
    '''