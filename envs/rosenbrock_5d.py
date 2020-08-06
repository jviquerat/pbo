# Generic imports
import math
import numpy as np

###############################################
### Environment for rosenbrock 5D function
class rosenbrock_5d():

    # Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'rosenbrock_5d'
        self.n_params = 5
        self.x_min    =-2.0
        self.x_max    = 2.0
        self.obs      = np.ones(1)

    ### Actual function
    def function(self, x):

        val = 0.0
        for i in range(len(x)-1):
            val += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2

        return -val

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions    = self.n_params*[None]
        x_scale         = 0.5*(self.x_max - self.x_min)
        conv_actions[:] = x_scale*actions[:]

        return conv_actions

    ### Take one step
    def step(self, actions, ep):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        reward       = self.function(conv_actions)

        return reward, conv_actions
