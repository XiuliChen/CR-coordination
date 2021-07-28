import gym
import os
from stable_baselines3 import PPO
from eye_hand_env import EyeHandEnv

from utils_plots import plot_learned_behaviour


# step1: parameter inputs
perceptual_noise=0.005
ocular_SDN,ocular_CN=0.0005,0.00005
motor_SDN,motor_CN=0.005,0.0005
pv_constant_hand,pv_slope_hand=40,2.5
fitts_D=0.5
fitts_W=0.2


# step2: create the logging folder
folder=f'../logs/D{fitts_D}W{fitts_W}'
perceptual_str=f'perceptual{perceptual_noise}'  
ocular_noise_str=f'ocular_SDN{ocular_SDN}ocular_CN{ocular_CN}'
motor_noise_str=f'motor_SDN{motor_SDN}motor_CN{motor_CN}'
pv_str=f'hand_constant={pv_constant_hand}hand_slope{pv_slope_hand}'

log_dir = f'{folder}{perceptual_str}{ocular_noise_str}{motor_noise_str}{pv_str}/'

log_dir=f'{log_dir}best_run/'

model=PPO.load(f'{log_dir}/ppo_model')


env= EyeHandEnv(fitts_W = fitts_W, fitts_D=fitts_D,
      perceptual_noise=perceptual_noise,
      ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
      motor_SDN=motor_SDN,motor_CN=motor_CN,
      pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand)

plots_dir=f'{log_dir}plots'
os.makedirs(plots_dir, exist_ok=True)
plot_learned_behaviour(model,plots_dir,env,10)