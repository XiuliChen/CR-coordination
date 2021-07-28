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
import numpy as np
import os
import shutil
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# local modules
from eye_hand_env import EyeHandEnv
from utils_plots import plot_learning_curve, plot_learned_behaviour


def main(perceptual_noise,ocular_SDN,ocular_CN,
      motor_SDN,motor_CN,pv_constant_hand,pv_slope_hand,step_cost,
      timesteps,log_dir,log_dir_run,train):
	
	# Instantiate the environment
	env0 = EyeHandEnv(perceptual_noise=perceptual_noise,
      ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
      motor_SDN=motor_SDN,motor_CN=motor_CN,
      pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand,
      step_cost=step_cost, train=train)

	# where to save the monitor data (e.g., to plot learning curve)
	env = Monitor(env0, log_dir_run)

	# Instantiate the model
	model = PPO('MlpPolicy', env, verbose=0)

	# Train the agent
	model.learn(total_timesteps=int(timesteps))

	# plot and save the learning curve
	plot_learning_curve(log_dir_run)
	plt.savefig(f'{log_dir_run}learning_curve.png')
	# save the final trained model
	model.save(f'{log_dir_run}saved_model/ppo_model')



	# Evaluate the trained agent
	n_eval_episodes=10000
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
	np.save(f'{log_dir_run}mean_reward.npy',np.array([mean_reward, std_reward]))

	# Save to the best-run folder if find a better run
	find_better_run=False
	max_reward_file=f'{log_dir}/best_run/max_reward.npy'
	if os.path.exists(max_reward_file):
		max_reward=np.load(max_reward_file)
		if mean_reward>max_reward[0]:
			find_better_run=True
	else:
		find_better_run=True

	if find_better_run:
		np.save(max_reward_file,np.array([mean_reward, std_reward]))
		model.save(f'{log_dir}/best_run/ppo_model')
		plt.savefig(f'{log_dir}/best_run/learning_curve.png')
		# copy the monitor data
		src = f'{log_dir_run}monitor.csv'
		dst =  f'{log_dir}best_run/monitor.csv'
		shutil.copyfile(src, dst)

		plots_dir=f'{log_dir}best_run/plots'
		os.makedirs(plots_dir, exist_ok=True)
		plot_learned_behaviour(model,plots_dir,env,20)







if __name__=='__main__':
	pass