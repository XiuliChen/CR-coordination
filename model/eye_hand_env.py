'''
By Xiuli Chen
23/July/2021 (new model)

Build the task as a Open AI gym environment
'''
import numpy as np
import gym
from gym import spaces


# local modules
from eye_model import EyeModel
from hand_model import HandModel
from vision_model import VisionModel

from utils import get_new_target, get_tgt_belief,calc_dis
from constants import *


class EyeHandEnv(gym.Env):
  #############################################################################
  def __init__(self,fitts_W,fitts_D, # task
    perceptual_noise, # vision
    ocular_SDN,ocular_CN, # ocular motor
    motor_SDN,motor_CN,pv_constant_hand,pv_slope_hand # hand model
    ):
    
    super(EyeHandEnv,self).__init__()

    ############ task setting ##################################################
    # the canvas of the display is [-1,1] for both x and y
    # unit 1.0 in the canvas equals to SCALE_DEG degress given in constants.py
    # For example, fitts_D=0.5, means that the target is SCALE_DEG*0.5 degree away
    self.fitts_W=fitts_W
    self.fitts_D=fitts_D

    ############ eye/hand/vision ###############################################
    self.eye_model=EyeModel(ocular_SDN,ocular_CN)
    self.hand_model=HandModel(motor_SDN,motor_CN,pv_constant_hand,pv_slope_hand)
    self.vision_model=VisionModel(perceptual_noise)
        
    #################### actions ###############################################
    # the action space and observation space
    self.actions=np.array([ [NO_OP,NO_OP],
                            [NEW_COMMAND,NO_OP],
                            [NO_OP,NEW_COMMAND],
                            [NO_OP,CLICK],
                            ])    
    self.action_space = spaces.Discrete(len(self.actions))

    #################### belief state ###########################################
    # see _state_observation function below
    low_b=np.array([-1.0, -1.0,# eye pos 
        -1.0, -1.0,# hand pos
        -1.0, -1.0,# tgt_belief
        0, # tgt_belief_uncertainty
        0, # tartget width
        -1.0,-1.0,# eye and hand stage
    ], dtype=np.float32)

    high_b=np.array([1.0, 1.0, # eye pos
      1.0, 1.0,# hand pos
      1.0, 1.0, # target pos
      0.5, #tgt_belief_uncertainty
      0.5, # target width
      1.0,1.0,# eye and hand stage
    ], dtype=np.float32)

    self.observation_space = spaces.Box(low=low_b, high=high_b, dtype=np.float32)

    #############################################################################
    self.max_steps=1000
    self.sim_time=50

    #############################################################################

  #############################################################################
  def _state_observation(self):
    self.state = np.concatenate((self.current_pos[EYE],
      self.current_pos[HAND],
      self.tgt_belief,
      self.tgt_belief_uncertainty,
      self.fitts_W,
      self.stage,
      ),axis=None)
    
    self.observation=self.state

  #############################################################################
  def reset(self):
    self.correct=0
    self.episode_steps=0

    # random target position
    self.target_pos = get_new_target(self.fitts_D)
    self.dis_to_target=self.fitts_D

    # The agent starts with fixating at the start position
    self.stage=np.array([FIXATE,STILL],dtype=np.float32)
    self.current_pos=np.zeros((2,2)) 
    self.current_vel=np.zeros(2)
    
    # This is used to monitor which step eye/hand is at the motor program
    self.progress_step=np.zeros(2, dtype=int)
    self.first=[True,True]
    self.eye_program ={'pos':np.zeros((1,2)), 'vel': np.zeros((1,1)), 'stage': FIXATE*np.ones((1,1),dtype=np.float32)}
    self.hand_program={'pos':np.zeros((1,2)), 'vel': np.zeros((1,1)), 'stage':  STILL*np.ones((1,1),dtype=np.float32)}
    
    # Initial observation and target position belief
    self.tgt_obs,self.tgt_obs_uncertainty=self.vision_model.get_tgt_obs(self.current_pos[EYE],self.target_pos)
    self.tgt_belief=self.tgt_obs
    self.tgt_belief_uncertainty=self.tgt_obs_uncertainty    
  
    # the state of the environment includes three elements: target, eye, hand
    self._state_observation()

    return self.observation

  #############################################################################
  def step(self,a):
    '''
    Execute the action, state transion and new observation
    return obs, reward, done, info
    '''
    # obs
    self.chosen_action=self.actions[a]
    self._state_transit(self.chosen_action)
    self._state_observation()

    if self.stage[HAND]==STILL:
      self.dis_to_target=calc_dis(self.target_pos, self.current_pos[HAND])

    # done and reward
    reward, done=self._reward_function(a)

    self.episode_steps+=1
    if self.episode_steps>self.max_steps:
      done=True

    return self.observation, reward, done, {}

  #############################################################################
  def _reward_function(self,action_index):
    '''
    r=F(s,a)
    '''
    step_cost=-0.01
    if action_index==ACTION_CLICK: # terminal action
      done=True
      if self.dis_to_target< self.fitts_W/2:
        reward=0
        self.correct=1
      else:
        reward=-1
    else: # non terminal actions
      done=False
      if action_index==ACTION_NOOP:
        reward=step_cost
      elif action_index==ACTION_NEW_EYE_COMMAND:
        reward=step_cost if self.stage[EYE]==FIXATE else step_cost*3
      elif action_index==ACTION_NEW_HAND_COMMAND:
        reward=step_cost if self.stage[HAND]==STILL else step_cost*3

    return reward, done

  #############################################################################
  def _state_transit(self,action):
    # update the state of eye, followed by hand.

    # the eye program cannot be interrupped
    if action[EYE]==NEW_COMMAND and self.stage[EYE]==FIXATE: 
      self._new_command(EYE)
    else: # carry on current motor program
      self._no_new_command(EYE)

    # generate new motor program
    if action[HAND]==NEW_COMMAND:  
      self._new_command(HAND)
    else: # carry on current motor program
      self._no_new_command(HAND)
  
  #############################################################################
  def _new_command(self,mode): 
    self.progress_step[mode]=0
    self.stage[mode]=PREP
    # generate new motor program towards current target belief
    aim_pos=self.tgt_belief
    
    stop_duration=50
    if self.first[mode]:
      prep_duration=150
    else:
      prep_duration=50

    if mode==EYE:
      pos, vel, stage, end_pos=self.eye_model.motor_program(self.current_pos[EYE],aim_pos,
        prep_duration,stop_duration)   
      self.eye_program['pos']=pos
      self.eye_program['vel']=vel
      self.eye_program['stage']=stage
    else: # HAND
      pos, vel, stage, end_pos=self.hand_model.motor_program(self.current_pos[HAND],aim_pos,
        prep_duration,stop_duration)
      self.hand_program['pos']=pos
      self.hand_program['vel']=vel
      self.hand_program['stage']=stage
  
  #############################################################################
  def _no_new_command(self,mode):
    # carry on the current process
    self.progress_step[mode]+=self.sim_time
    if mode==EYE:
      program=self.eye_program  
    else:
      program=self.hand_program

    iProgress=self.progress_step[mode]
    if iProgress>=len(program["stage"]):     
      iProgress=-1

    # update the stage
    previous_stage=self.stage[mode] 
    self.stage[mode]=program["stage"][iProgress]
    self.current_pos[mode]=program["pos"][iProgress]
    self.current_vel[mode]=program["vel"][iProgress]
    
    '''
    if previous_stage==PREP and self.stage[mode]!=PREP:
      # complete prep of the first submovement
      self.first[mode]=False
    '''
    # the eyes are able to collect more info if they are not moving
    # i.e., still receive info when preping
    if self.stage[EYE]!=RAMPUP and self.stage[EYE]!=RAMPDOWN:
      self.tgt_obs,self.tgt_obs_uncertainty=self.vision_model.get_tgt_obs(self.current_pos[EYE],self.target_pos)
      self.tgt_belief,self.tgt_belief_uncertainty=get_tgt_belief(self.tgt_obs,self.tgt_obs_uncertainty,
        self.tgt_belief,self.tgt_belief_uncertainty)


#############################################################################
if __name__=="__main__":
  from utils_plots import *
  # UNIT TEST
  # THIS IS TO TEST ONE TRAIL IN THE MODEL. 
  render=True
  if render:
    plt.figure(figsize=(15,15))

  perceptual_noise=0.09
  ocular_SDN=0.01
  ocular_CN=0.001

  motor_SDN=0.01
  motor_CN=0.001
  pv_constant_hand=40
  pv_slope_hand=2.5
  timesteps=1e6

  fitts_W=0.02
  fitts_D=0.5

  # Instantiate the environment
  env = EyeHandEnv(fitts_W = fitts_W, fitts_D=fitts_D,
      perceptual_noise=perceptual_noise,
      ocular_SDN=ocular_SDN,ocular_CN=ocular_CN,
      motor_SDN=motor_SDN,motor_CN=motor_CN,
      pv_constant_hand=pv_constant_hand,pv_slope_hand=pv_slope_hand)


  obs=env.reset()
  done = False
  t=0
  while not done:
    t+=1
    if render:
      print(np.round(obs,3))
      plot_episode(t,env)

    # choose an action 
    if env.stage[EYE]==FIXATE:
      action=ACTION_NEW_EYE_COMMAND #[NEW_COMMAND,NO_OP]
      print('NEW_EYE_COMMAND')
    elif env.stage[HAND]==STILL:
      action=ACTION_NEW_HAND_COMMAND # [NO_OP,NEW_COMMAND]
      print('NEW_HAND_COMMAND')
    else:
      action=ACTION_NOOP #[NO_OP,NO_OP]
      print('NOOP')

    if env.dis_to_target<env.fitts_W/2:
      action=ACTION_CLICK
      print('CLICK')

    # one step further given the step
    obs, reward, done, info=env.step(action)
    

  if render:
    plt.savefig('figures/EyeHandTest')

