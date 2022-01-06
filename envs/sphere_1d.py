# Generic imports
import math
import numpy as np

# Custom imports
from envs.base_env import base_env

###############################################
### Environment for sphere_1d function
class sphere_1d(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'sphere_1d'
        self.act_size = 1
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([-5.0])
        self.x_max    = np.array([ 5.0])
        self.x_0      = np.array([ 4.0])

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = 0.0
        for i in range(len(x)):
            val += (x[i])**2

        return -val
