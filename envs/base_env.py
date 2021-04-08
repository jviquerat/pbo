# Generic imports
import math
import numpy as np

###############################################
### Base environment
class base_env():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'base_env'
        self.act_size = 100
        self.obs_size = self.act_size
        self.obs      = np.ones(self.obs_size)
        self.x_min    =-100.0
        self.x_max    = 100.0

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions  = self.act_size*[None]
        x_p           = self.x_max - self.x_0
        x_m           = self.x_0   - self.x_min

        for i in range(self.act_size):
            if (actions[i] >= 0.0):
                conv_actions[i] = self.x_0[i] + x_p[i]*actions[i]
            if (actions[i] <  0.0):
                conv_actions[i] = self.x_0[i] + x_m[i]*actions[i]

        return conv_actions

    ### Take one step
    def step(self, actions, ep):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        reward       = self.function(conv_actions)

        return reward, conv_actions

    ### Close environment
    def close(self):
        pass
