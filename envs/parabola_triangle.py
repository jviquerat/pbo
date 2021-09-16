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
        self.x_min    = np.array([0.0,  1.0])
        self.x_max    = np.array([0.0,  1.0])
        self.x_0      = np.array([0.35, 0.35])

    ### Convert actions
    def convert_actions(self, actions):

        # Initialize conv_actions array
        conv_actions = self.act_size*[None]

        # Map first component to [0,1]
        conv_actions[0] = 0.5*(actions[0]+1.0)

        # Map second component to [0,1-x]
        conv_actions[1] = 0.5*(actions[1]+1.0)*(1.0-conv_actions[0])

        return conv_actions

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = (x[0] - 0.1)**2 + (x[1] - 0.8)**2

        return -val
