import numpy as np
import matplotlib.pyplot as plt
from constants import *


def plot_program(current_pos,aim_pos,end_pos,pos,vel,scale_deg,stage):
    current_pos=np.array(current_pos)*scale_deg
    aim_pos=np.array(aim_pos)*scale_deg
    end_pos=np.array(end_pos)*scale_deg
    pos=np.array(pos)*scale_deg

    print('Figure 1: given current_pos and aim_pos, the hand model generates the trajectory' )
    plt.figure(figsize=(15,15))

    plt.subplot(4,1,1)
    plt.plot(current_pos[0],current_pos[1],'g>',label='start',markersize=15)
    plt.plot(aim_pos[0],aim_pos[1],'r+',label='aim',markersize=15)
    plt.text(aim_pos[0],aim_pos[1],f'{np.round(aim_pos,2)}')
    plt.plot(end_pos[0],end_pos[1],'b*',label='aim+noise',markersize=15)
    plt.text(end_pos[0],end_pos[1],f'{np.round(end_pos,2)}')
    
    for i in range(len(stage)):
        if stage[i]==PREP:
            plt.plot(pos[i][0],pos[i][1],'g>',markersize=15)
        elif stage[i]==MOVING:
            plt.plot(pos[i][0],pos[i][1],'m.')
        elif stage[i]==FIXATE:

            plt.plot(pos[i][0],pos[i][1],'b*')

    plt.legend()
    plt.xlabel('visual angle(degree)')
    plt.ylabel('visual angle(degree)')
    plt.title('task')


    plt.subplot(4,1,2)
    plt.plot(-5,pos[0][0],'g.',label='prep')
    plt.plot(-5,pos[0][0],'m.',label='move')
    plt.plot(-5,pos[0][0],'b*',label='stop')
    for i in range(len(stage)):
        if stage[i]==PREP:
            plt.plot(i,pos[i][0],'g>')
        elif stage[i]==MOVING:
            plt.plot(i,pos[i][0],'m.')
        elif stage[i]==FIXATE:
            plt.plot(i,pos[i][0],'b*')

    plt.legend()
    plt.xlim(-1,i+1)
    plt.xlabel(f'time(ms')
    plt.ylabel('x')


    plt.subplot(4,1,3)

    for i in range(len(stage)):
        if stage[i]==PREP:
            plt.plot(i,pos[i][1],'g>')
        elif stage[i]==MOVING:
            plt.plot(i,pos[i][1],'m.')
        elif stage[i]==FIXATE:
            plt.plot(i,pos[i][1],'b*')
    plt.xlabel(f'time(ms)')
    plt.ylabel('y')
    plt.xlim(-1,i+1)


    plt.subplot(4,1,4)

    for i in range(len(stage)):
        if stage[i]==PREP:
            plt.plot(i,vel[i],'g>')
        elif stage[i]==MOVING:
            plt.plot(i,vel[i],'m.')
        elif stage[i]==FIXATE:
            plt.plot(i,vel[i],'b*')

    plt.xlabel(f'time(ms)')
    plt.ylabel('vel (deg/s)')
    plt.xlim(-1,i+1)

    plt.suptitle('Given star_pos and aim_pos, the hand model generates the trajectory between start and aim+noise')
    plt.savefig('figures/HandModelTest.png')

