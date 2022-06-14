# Generic imports
import os
import math
import numpy as     np
from   math  import tanh

# Custom imports
from  base_env import base_env

###############################################
### Base environment for cartpole control
class cartpole(base_env):

    # Create object
    def __init__(self, path):

        # Problem parameters
        self.gravity         = 9.8
        self.masscart        = 1.0
        self.masspole        = 0.1
        self.total_mass      = self.masspole + self.masscart
        self.length          = 0.5
        self.polemass_length = self.masspole*self.length
        self.force_mag       = 10.0
        self.tau             = 0.02
        self.t_max           = 20.0

        # Angle at which to fail the episode
        self.th_threshold_radians = 12.0*2.0*math.pi/360.0
        self.x_threshold          = 2.4

        # Environment parameters
        self.act_size = 7
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.ones(self.act_size)*(-1.0)
        self.x_max    = np.ones(self.act_size)*( 1.0)
        self.x_0      = np.ones(self.act_size)*( 0.0)

        # Set path
        self.path = path
        os.makedirs(path, exist_ok=True)

    # Reset variables
    def reset(self):

        self.x      = np.random.uniform(low=-0.05, high=0.05)
        self.x_dot  = np.random.uniform(low=-0.05, high=0.05)
        self.th     = np.random.uniform(low=-0.05, high=0.05)
        self.th_dot = np.random.uniform(low=-0.05, high=0.05)
        self.x_acc  = 0.0
        self.th_acc = 0.0

        self.lst_t      = []
        self.lst_x      = []
        self.lst_x_dot  = []
        self.lst_x_acc  = []
        self.lst_th     = []
        self.lst_th_dot = []
        self.lst_th_acc = []
        self.lst_u      = []

    ### Take one step
    def step(self, actions, ep):

        # Convert actions
        cv_act = self.convert_actions(actions)

        # Reset all
        self.reset()

        # Integrate
        self.integrate(cv_act)

        # Compute reward
        rwd = self.get_rwd()

        # Dump data
        self.dump(ep)

        return rwd, cv_act

    # Integrate system
    def integrate(self, act):

        t = 0.0

        while (t < self.t_max):
            u = self.ctrl(act)*self.force_mag

            costh = math.cos(self.th)
            sinth = math.sin(self.th)

            temp         = (u + self.polemass_length*self.th_dot**2*sinth)
            temp        /= self.total_mass
            self.th_acc  = (self.gravity*sinth - costh*temp)
            self.th_acc /= self.length*(4.0/3.0-self.masspole*costh**2/self.total_mass)
            self.x_acc   = temp-self.polemass_length*self.th_acc*costh
            self.x_acc  /= self.total_mass

            self.x      += self.tau*self.x_dot
            self.x_dot  += self.tau*self.x_acc
            self.th     += self.tau*self.th_dot
            self.th_dot += self.tau*self.th_acc

            t += self.tau

            self.lst_t.append(t)
            self.lst_x.append(self.x)
            self.lst_x_dot.append(self.x_dot)
            self.lst_x_acc.append(self.x_acc)
            self.lst_th.append(self.th)
            self.lst_th_dot.append(self.th_dot)
            self.lst_th_acc.append(self.th_acc)
            self.lst_u.append(u)

    # Control law
    def ctrl(self, act):

        # u = tanh(act[0]*self.x +
        #          act[1]*self.x_dot +
        #          act[2]*self.th +
        #          act[3]*self.th_dot +
        #          act[4])
        u = tanh(act[0]*self.x +
                 act[1]*self.x_dot +
                 act[2]*self.x_acc +
                 act[3]*self.th +
                 act[4]*self.th_dot +
                 act[5]*self.th_acc +
                 act[6])

        return u

    # Compute reward
    def get_rwd(self):

        rwd = 0.0
        for i in range(len(self.lst_x)-1):
            x      = self.lst_x[i]
            x_dot  = self.lst_x_dot[i]
            th     = self.lst_th[i]
            th_dot = self.lst_th_dot[i]
            u      = self.lst_u[i]

            # First type of reward
            if (    x  > -self.x_threshold
                and x  <  self.x_threshold
                and th > -self.th_threshold_radians
                and th <  self.th_threshold_radians):
                rwd += 1.0
                rwd -= 0.05*abs(u)
            else:
                rwd -= 1.0

        return rwd

    # Dump
    def dump(self, ep):

        filename = self.path+'/'+str(ep)+'.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.lst_t,      (-1,1)),
                              np.reshape(self.lst_x,      (-1,1)),
                              np.reshape(self.lst_x_dot,  (-1,1)),
                              np.reshape(self.lst_th,     (-1,1)),
                              np.reshape(self.lst_th_dot, (-1,1)),
                              np.reshape(self.lst_u,      (-1,1))]),
                   fmt='%.5e')
