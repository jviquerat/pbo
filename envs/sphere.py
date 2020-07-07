# Generic imports
import math
import numpy as np

###############################################
### Environment for sphere function
class sphere():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'sphere'
        self.n_params = 3
        self.x_min    =-5.0
        self.x_max    = 5.0
        self.y_min    =-5.0
        self.y_max    = 5.0
        self.z_min    =-5.0
        self.z_max    = 5.0
        self.obs      = np.ones(1)

    ### Actual function
    def function(self, x, y, z):

        return -(x**2 + y**2 + z**2)

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions    = self.n_params*[None]
        x_scale         = 0.5*(self.x_max - self.x_min)
        y_scale         = 0.5*(self.y_max - self.y_min)
        z_scale         = 0.5*(self.z_max - self.z_min)
        conv_actions[0] = x_scale*actions[0]
        conv_actions[1] = y_scale*actions[1]
        conv_actions[2] = z_scale*actions[2]

        return conv_actions

    ### Take one step
    def step(self, actions, ep):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        x            = conv_actions[0]
        y            = conv_actions[1]
        z            = conv_actions[2]
        reward       = self.function(x,y,z)

        return reward, conv_actions
