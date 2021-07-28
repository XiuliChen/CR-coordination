import gym
import os
from stable_baselines3 import PPO
from eye_hand_env import EyeHandEnv
import matplotlib.pyplot as plt
import numpy as np
from utils_plots import plot_learned_behaviour,MT_learned_behaviour


# step1: parameter inputs
perceptual_noise=0.005
ocular_SDN,ocular_CN=0.0005,0.00005
motor_SDN,motor_CN=0.005,0.0005
pv_constant_hand,pv_slope_hand=40,2.5
step_cost=-0.01


# step2: create the logging folder
folder=f'../logs/all_targets/'
perceptual_str=f'perceptual{perceptual_noise}'  
ocular_noise_str=f'ocular_SDN{ocular_SDN}ocular_CN{ocular_CN}'
motor_noise_str=f'motor_SDN{motor_SDN}motor_CN{motor_CN}'
pv_str=f'hand_constant={pv_constant_hand}hand_slope{pv_slope_hand}'
step_cost_str=f'step_cost{step_cost}'

log_dir = f'{folder}{perceptual_str}{ocular_noise_str}{motor_noise_str}{pv_str}{step_cost_str}/'


log_dir=f'{log_dir}best_run/'

model=PPO.load(f'{log_dir}/ppo_model')



exp=4
Ws,Ds,exp_str=experiments(exp)
shapes=['^:','o:','s:','>:','*:']
colors=['#d73027',
'#91bfdb',
'#4575b4','#fee090','#fc8d59',]

for cw,fitts_W in enumerate(Ws):
      mt_w=[]
      for cd,fitts_D in enumerate(Ds):
            env= EyeHandEnv(
                  perceptual_noise=perceptual_noise,
                  ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
                  motor_SDN=motor_SDN,motor_CN=motor_CN,
                  pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand,
                  step_cost=step_cost, train=False,fitts_W = fitts_W, fitts_D=fitts_D)

            mean_mt,std_mt,acc=MT_learned_behaviour(model,env,1000)
            
            plt.subplot(1,2,2)
            plt.plot(cw*len(Ds)+cd+1, acc,shapes[cw], color=colors[cd], label=f'W={fitts_W}D={fitts_D}')
            mt_w.append(mean_mt)
      plt.subplot(1,2,1)
      plt.plot(Ds,mt_w,shapes[cw])

plt.subplot(1,2,1)
plt.title('MT')
plt.subplot(1,2,2)
plt.title('acc')
plt.legend()
plt.savefig(f'{exp_str}_{perceptual_str}{ocular_noise_str}{motor_noise_str}{pv_str}{step_cost_str}.png')


