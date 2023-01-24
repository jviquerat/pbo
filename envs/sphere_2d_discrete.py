# Generic imports
import math
import numpy as np

# Custom imports
from envs.base_env import base_env

###############################################
### A dummy descrete environment for descrete pbo testing
class sphere_2d_discrete(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'sphere_2d_discrete'
        self.x_values = [np.linspace(-5,5,51), np.linspace(-5,5,51)]
        self.act_size = [len(val) for val in self.x_values] #act_size becomes a list here (for each parameter, it gives the number of possibilities)
        self.obs_size = len(self.act_size)
        self.obs      = np.zeros(self.obs_size)

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = 0.0
        for i in range(len(x)):
            val += (x[i])**2

        return -val

    def convert_actions(self, actions): #Overwrite for descrete actions: actions is a set of indices in this case
        
        conv_actions  = len(actions)*[None]
        for i in range(len(actions)):
            conv_actions[i] = self.x_values[i][int(actions[i])]

        return conv_actions
