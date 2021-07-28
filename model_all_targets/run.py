
from main import main
import os
import numpy as np

from experiment import experiments

import argparse
my_parser=argparse.ArgumentParser()
my_parser.add_argument('--n',help='run index', action='store',type=int)
args=my_parser.parse_args()

'''
# step1: parameter inputs
# step2: create the logging folder
# step3: call the main
'''
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
log_dir_run=f'{log_dir}run{args.n}/'

os.makedirs(log_dir_run, exist_ok=True)
os.makedirs(f'{log_dir_run}/saved_model/', exist_ok=True)
os.makedirs(f'{log_dir}/best_run/', exist_ok=True)


# step3: call the main
timesteps=2e6
main(perceptual_noise=perceptual_noise,
  ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
  motor_SDN=motor_SDN,motor_CN=motor_CN,
  pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand,step_cost=step_cost,
  timesteps=timesteps,log_dir=log_dir,log_dir_run=log_dir_run,train=True)




