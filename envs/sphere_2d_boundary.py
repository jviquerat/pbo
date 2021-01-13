# Generic imports
import math
import numpy as np

# Custom imports
from envs.base_env import base_env

###############################################
### Environment for sphere_2d function
class sphere_2d_boundary(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'sphere_2d_boundary'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([-5.0,-5.0])
        self.x_max    = np.array([ 5.0, 5.0])
        self.x_0      = np.array([ 0.0, 0.0])

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = 0.0
        for i in range(len(x)):
            val += (x[i]+5.0)**2

        return -val
