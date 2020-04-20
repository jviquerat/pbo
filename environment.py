# Generic imports
import os
import math
import gym
import numpy as     np
from   gym   import spaces

###############################################
### Define gym environment
###############################################

###############################################
### Class env_opt
### A gym environment for optimization
class env_opt(gym.Env):

    ### Create object
    def __init__(self, init_obs, n_params, x_min, x_max, y_min, y_max):

        # Init from parent class
        super(env_opt, self).__init__()

        # Fill structure
        self.rank           = 0
        time                = 'now'
        path                = '.'
        self.path           = path+'_'+time+'/'
        self.output_path    = self.path+str(self.rank)+'/'
        self.reward_path    = self.output_path
        self.action_path    = self.output_path
        self.state_path     = self.output_path
        self.actions        = None
        self.conv_actions   = None
        self.last_actions   = None
        self.states         = None
        self.reward         = 0.0
        self.n_params       = n_params
        self.init_obs       = init_obs
        self.x_min          = x_min
        self.x_max          = x_max
        self.y_min          = y_min
        self.y_max          = y_max

        # Define output reward folder
        if (not os.path.exists(self.reward_path)):
            os.makedirs(self.reward_path)

        # Define output action folder
        if (not os.path.exists(self.action_path)):
            os.makedirs(self.action_path)

        # Define output state folder
        if (not os.path.exists(self.state_path)):
            os.makedirs(self.state_path)

        # Define action space
        self.action_size  = self.n_params
        self.action_space = spaces.Box(low   =-1.0,
                                       high  = 1.0,
                                       shape = (self.action_size,),
                                       dtype = np.float16)

        # Define observation space
        self.state_size = self.n_params
        self.observation_space = spaces.Box(low   =-1.0,
                                            high  = 1.0,
                                            shape = (self.state_size,),
                                            dtype = np.float16)

        # Set problem name
        self.episode      =-1
        self.totalstep    = 0

    ### Provide next observation
    def next_observation(self):

        # Just an array of ones
        obs = self.init_obs

        # Fill states for file saving
        self.states = obs

        return obs

    ### Choose next action
    def take_action(self, actions):

        # Reset action array
        self.actions = []
        self.conv_actions = self.action_size*[None]

        # Convert actions
        self.last_actions = actions
        if (actions[0] < 0.0):
            self.conv_actions[0] = abs(self.x_min)*actions[0]
        else:
            self.conv_actions[0] = self.x_max*actions[0]
        if (actions[1] < 0.0):
            self.conv_actions[1] = abs(self.y_min)*actions[1]
        else:
            self.conv_actions[1] = self.y_max*actions[1]

        # Store
        self.actions = np.append(self.actions, [actions, self.conv_actions])

    ### Take one step
    def step(self, actions):

        # Take action and solve new problem
        self.take_action(actions)

        # Compute reward
        self.reward = self.compute_reward(self.conv_actions)

        # Collect observations
        obs = self.next_observation()

        # Update step index
        self.totalstep += 1

        # Episode end
        done = True

        return self.reward, self.conv_actions

    ### Compute reward
    def compute_reward(self, x):

        # Return function value
        return -(x[0]**2 + x[1]**2)
        #return -((1.0-x[0])**2 + 100.0*(x[1]-x[0]**2)**2)

    ### Reset environment
    def reset(self):

        # Increment episode
        self.episode += 1

        # Provide initial observations
        return self.next_observation()

    ### Rendering
    def render(self, mode='human', close=False):

        # Empty for now
        return 0
