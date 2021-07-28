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
        elif stage[i]==RAMPUP or stage[i]==RAMPDOWN:
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
        elif stage[i]==RAMPUP or stage[i]==RAMPDOWN:
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
        elif stage[i]==RAMPUP or stage[i]==RAMPDOWN:
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
        elif stage[i]==RAMPUP or stage[i]==RAMPDOWN:
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



###########################################################
# These are for plotting the learning curve
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_learning_curve(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'last average (window=100)= {np.round(y[-1],3)}')   


def plot_learned_behaviour(model,log_dir,env,n_eps):
    '''
    run the trained model, return the behaviours on each step
    '''
    eps=0
    markersize=7
    offset=0.01
    while eps<n_eps:
        obs=env.reset()
        done=False

        plt.close()
        plt.figure(figsize=(7,7))

        dis_hand_all=[]
        dis_eye_all=[]
        dis_eye_all.append(env.fitts_D)
        dis_hand_all.append(env.fitts_D)

        while not done:
            #action, _ = model.predict(obs,deterministic = True)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            #plot_learned_episode(env,done,action)
            dis_eye=calc_dis(env.target_pos,env.current_pos[EYE])
            dis_hand=calc_dis(env.target_pos,env.current_pos[HAND])

            dis_eye_all.append(dis_eye)
            dis_hand_all.append(dis_hand+offset*HAND)
            

            # plot eye and then hand

            if action==ACTION_CLICK:
                plt.plot(env.episode_steps,dis_hand,'*',markersize=markersize, color='r')
            else:
                if action==ACTION_NOOP:
                    plt.plot(env.episode_steps,0.3,'.',markersize=markersize,color='k',
                            markerfacecolor='b')

                elif action==ACTION_NEW_EYE_COMMAND:
                    plt.plot(env.episode_steps,0.3,'o',markersize=markersize,color='k',
                            markerfacecolor='m')
                elif action==ACTION_NEW_HAND_COMMAND:
                    plt.plot(env.episode_steps,0.3,'>',markersize=markersize,color='k',
                            markerfacecolor='c')


                for mode in [EYE,HAND]:                  
                    if mode==EYE:
                        dis=dis_eye
                        cc='k'
                    else:
                        dis=dis_hand
                        cc='r'

                    if env.actions[action][mode]==NEW_COMMAND:
                        plt.plot(env.episode_steps,dis+offset*mode,'o',markersize=markersize, color=cc,
                            markerfacecolor=cc)
                    else:

                        if env.stage[mode]==FIXATE:
                            plt.plot(env.episode_steps,dis+offset*mode,'s',markersize=markersize, color=cc,
                                markerfacecolor=cc)      
                        elif env.stage[mode]==PREP:
                            plt.plot(env.episode_steps,dis+offset*mode,'>',markersize=markersize, color=cc,
                                markerfacecolor=cc)
                        elif env.stage[mode]==RAMPUP or env.stage[mode]==RAMPDOWN:
                            plt.plot(env.episode_steps,dis+offset*mode,'.',markersize=markersize, color=cc,
                                markerfacecolor=cc)



            if done:


                plt.hlines(0,0,env.episode_steps,colors='b',linestyles='-')
                plt.hlines(env.fitts_W/2,0,env.episode_steps,colors='g',linestyles='--',label='Target region')

                plt.plot(-10,dis_eye,'ks',markersize=markersize,label='Eye fixating')  
                plt.plot(-10,dis_eye,'ko',markersize=markersize, markerfacecolor='k', label='new command')      
                plt.plot(-11,dis_eye,'k>',markersize=markersize,label='Eye preping')


                plt.plot(-10,dis_hand,'rs',markersize=markersize,label='Hand still')
                plt.plot(-11,dis_hand,'r>',markersize=markersize,label='Hand preping')
                plt.plot(-10,dis_hand,'ro',markersize=markersize, markerfacecolor='r', label='new command')


                plt.plot(-10,dis_hand,'r*',markersize=markersize,label='Hand click')

                plt.plot(dis_eye_all,'k:')
                plt.plot(dis_hand_all,'r:')

                plt.xlim(-1,env.episode_steps+1)
                plt.xticks(np.arange(0,env.episode_steps+1,1),rotation='vertical')
                plt.xlabel(f'time ({env.sim_time}ms)')
                plt.ylabel('distance to target')
                if env.correct==1:
                    plt.title('hit target!')
                else:
                    plt.title('miss target!')
                plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper right')
                plt.tight_layout()
                plt.grid()

                plt.savefig(f'{log_dir}/e{eps}.png')
                eps+=1



    