# Custom imports
from envs.lorenz import *

###############################################
### Environment for Lorenz stabilizer
class lorenz_stabilizer(lorenz):

    # Create object
    def __init__(self, path):
        super().__init__(path)

        # Fill structure
        self.name = 'lorenz_stabilizer'

    # Compute reward
    def get_rwd(self):

        rwd = 0.0
        for i in range(len(self.lst_x)):
            if (self.lst_x[i] < 0.0): rwd += 1.0*self.dt

        return rwd
