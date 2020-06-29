# Generic imports
import math
import numpy as np

###############################################
### Environment for parabola function
class parabola():

    ### Create object
    def __init__(self):

        # Fill structure
        self.name     = 'parabola'
        self.n_params = 2
        self.x_min    =-5.0
        self.x_max    = 5.0
        self.y_min    =-5.0
        self.y_max    = 5.0
        self.obs      = np.ones(1)

    ### Actual function
    def function(self, x, y):

        return -(x**2 + y**2)

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
        conv_actions[0] = x_scale*actions[0] + 3.0
        conv_actions[1] = y_scale*actions[1] + 3.0

        return conv_actions

    ### Take one step
    def step(self, actions):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        x            = conv_actions[0]
        y            = conv_actions[1]
        reward       = self.function(x,y)

        return reward, conv_actions
