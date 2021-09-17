# Generic imports
import math
import numpy as np

# Custom imports
from envs.base_env import base_env

###############################################
### Environment for parabola_triangle function
class parabola_triangle(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'parabola_triangle'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_0      = np.array([0.2, 0.2])

    ### Convert actions
    def convert_actions(self, actions):

        # Initialize arrays
        conv_actions = self.act_size*[None]
        x_p          = self.act_size*[None]
        x_m          = self.act_size*[None]

        # Convert second component
        x_p[0] = 1.0 - self.x_0[0]
        x_m[0] = self.x_0[0] - 0.0

        if (actions[0] >= 0.0):
            conv_actions[0] = self.x_0[0] + x_p[0]*actions[0]
        if (actions[0] <  0.0):
            conv_actions[0] = self.x_0[0] + x_m[0]*actions[0]

        x_p[1] = 1.0 - conv_actions[0] - self.x_0[1]
        x_m[1] = self.x_0[1] - 0.0

        if (actions[1] >= 0.0):
            conv_actions[1] = self.x_0[1] + x_p[1]*actions[1]
        if (actions[1] <  0.0):
            conv_actions[1] = self.x_0[1] + x_m[1]*actions[1]

        #conv_actions[0] = 0.5*(actions[0]+1.0)
        #conv_actions[1] = 0.5*(actions[1]+1.0)*(1.0-conv_actions[0])

        return conv_actions

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = (x[0] - 0.1)**2 + (x[1] - 0.8)**2

        return -val
