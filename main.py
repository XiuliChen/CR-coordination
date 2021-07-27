'''
By Xiuli Chen
Last update 16/Dec/2020
STEPS:
#1. Instantiate the task environment
#2. Instantiate the model
#3. Train the model
#4. save the model
#5. Plot and save the learning curve
#6. Evaluate the model
'''

import os
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# local modules
from eye_hand_env import EyeHandEnv


def main(fitts_W, fitts_D,
      perceptual_noise,
      ocular_SDN,ocular_CN,
      motor_SDN,motor_CN,
      pv_constant_hand,pv_slope_hand,
      timesteps,log_dir):
	
	# Instantiate the environment
	env0 = EyeHandEnv(fitts_W = fitts_W, fitts_D=fitts_D,
      perceptual_noise=perceptual_noise,
      ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
      motor_SDN=motor_SDN,motor_CN=motor_CN,
      pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand)
	



	env = Monitor(env0, log_dir)

	# Instantiate the model
	model = PPO('MlpPolicy', env, verbose=1)

	# Train the agent
	model.learn(total_timesteps=int(timesteps))

	# save the final trained model
	model.save(f'{log_dir}saved_model/ppo_model')

	# Plot and save the learning curve
	plot_learning_curve(log_dir)
	plt.savefig(f'{log_dir}learning_curve.png')

	# Evaluate the trained agent
	n_eval_episodes=10000
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
	np.save(f'{log_dir}mean_reward.npy',np.array([mean_reward, std_reward]))



if __name__=='__main__':
	'''
	# step1: parameter inputs
	# step2: create the logging folder
	# step3: call the main
	'''

	# step1: parameter inputs
	perceptual_noise=0.09
	ocular_SDN=0.01
	ocular_CN=0.001

	motor_SDN=0.01
	motor_CN=0.001
	pv_constant_hand=40
	pv_slope_hand=2.5
	timesteps=1e6

	fitts_W=0.01
	fitts_D=0.5

	# step2: create the logging folder
	folder=f'logs/D{fitts_D}W{fitts_W}/'
	motor_noise_str=f'motor_SDN{motor_SDN}motor_CN{motor_CN}'
	ocular_noise_str=f'ocular_SDN{ocular_SDN}ocular_CN{ocular_CN}'
	log_dir = f'{folder}{motor_noise_str}{ocular_noise_str}/'

	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(f'{log_dir}plots/', exist_ok=True)
	os.makedirs(f'{log_dir}saved_model/', exist_ok=True)

	# step3: call the main

	# Instantiate the environment
	main(fitts_W = fitts_W, fitts_D=fitts_D,
      perceptual_noise=perceptual_noise,
      ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
      motor_SDN=motor_SDN,motor_CN=motor_CN,
      pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand,
      timesteps=timesteps,log_dir=log_dir)



