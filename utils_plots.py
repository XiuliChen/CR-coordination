import numpy as np
import matplotlib.pyplot as plt
from constants import *
from utils import calc_dis

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

def plot_episode(t,env):
  
  if t==1:
    plt.subplot(221)
    plt.plot(0,0,'gs',label='start')
    plt.plot(env.target_pos[0],env.target_pos[1],'r*',label='target') 
    #plt.plot(obs[0],obs[1],'b+',label='estimate')
    plt.plot(env.current_pos[EYE][0],env.current_pos[EYE][1],'ko',label='eye')
    plt.plot(env.current_pos[HAND][0],env.current_pos[HAND][1],'m.',label='hand')
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)
    plt.legend()




  plt.subplot(222)
  if env.stage[EYE]==FIXATE:
    plt.plot(t,env.current_pos[EYE][0],'ko',markerfacecolor='w',label='eye')
  else:
    plt.plot(t,env.current_pos[EYE][0],'k.',label='eye')

  if env.stage[HAND]==FIXATE:
    plt.plot(t,env.current_pos[HAND][0],'mo',markerfacecolor='w',label='hand')
  else:
    plt.plot(t,env.current_pos[HAND][0],'m.',label='hand')

  plt.hlines(env.target_pos[0],0,t,color='b')
  
  plt.subplot(223)
  if env.stage[EYE]==FIXATE:
    plt.plot(t,env.current_pos[EYE][1],'ko',markerfacecolor='w',label='eye')
  else:
    plt.plot(t,env.current_pos[EYE][1],'k.',label='eye')

  if env.stage[HAND]==FIXATE:
    plt.plot(t,env.current_pos[HAND][1],'mo',markerfacecolor='w',label='hand')
  else:
    plt.plot(t,env.current_pos[HAND][1],'m.',label='hand')

  plt.hlines(env.target_pos[1],0,t,color='b')

  plt.subplot(224)
  dis_eye=env.fitts_D-calc_dis(env.target_pos,env.current_pos[EYE])
  if env.stage[EYE]==FIXATE:
    plt.plot(t,dis_eye,'ko',markerfacecolor='w',label='eye')
  else:
    plt.plot(t,dis_eye,'k.',label='eye')

  dis_hand=env.fitts_D-calc_dis(env.target_pos,env.current_pos[HAND])
  if env.stage[HAND]==FIXATE:
    plt.plot(t,dis_hand,'mo',markerfacecolor='w',label='hand')
  else:
    plt.plot(t,dis_hand,'m.',label='hand')

  plt.hlines(env.fitts_D,0,t,color='b')



   

