# Generic imports
import numpy as np
import math

# Custom imports
from utils.shapes          import *
from utils.ctrl_cyl_solver import *
from envs.base_env         import *

###############################################
### Environment for control cylinder
class ctrl_cyl_1(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'ctrl_cyl_env'
        self.n_cyl    = 1
        self.act_size = 2*self.n_cyl
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.path     = path

        self.x_min    =-0.1
        self.x_max    = 0.5
        self.y_min    =-0.1
        self.y_max    = 0.1
        self.Re       = 100.0
        self.dy       = 0.00125
        self.ny       = math.floor((self.y_max-self.y_min)/self.dy)

        self.obs_diam = [0.025,0.01]
        self.obs_pos  = [[0.0,0.0],[1.0,0.0]]

        # self.rx_min   = (0.5*self.obs_diam[0]+0.5*self.obs_diam[1])*1.414
        # self.rx_max   = self.r_min + 10.0*self.obs_diam[1]
        # self.ry_min   = (0.5*self.obs_diam[0]+0.5*self.obs_diam[1])*1.414
        # self.ry_max   = self.r_min + 10.0*self.obs_diam[1]
        self.rx_min   = 0.025+2*0.01
        self.rx_max   = 0.08
        self.ry_min   = 0.025+2*0.01
        self.ry_max   = 0.08
        self.t_min    = 0.0
        self.t_max    = math.pi

    ### Take one step
    def step(self, act, ep):

        # Convert actions
        acc = self.convert_actions(act)

        # Fill converted actions
        for i in range(self.n_cyl):
            self.obs_pos[i+1] = acc[i]

        # Generate new shape
        drag, success = cfd_solve(self.x_min,    self.x_max,
                                  self.y_min,    self.y_max,
                                  self.ny,       self.Re,
                                  self.obs_diam, self.obs_pos,
                                  self.path,     str(ep))

        # Compute reward
        rwd = -abs(drag)

        return rwd, acc

    ### Convert actions
    def convert_actions(self, flat_act):

        # Make new array for new cylinder positions
        act      = np.array(flat_act).reshape((self.n_cyl,2))
        acc      = act.copy()
        acc[:,:] = 0.0

        # Convert action to coordinates
        for i in range(self.n_cyl):
            acc[i,:] = self.geom_transform(act[i,:])

        return acc

    ### Compute forward geometrical transformation for actions
    ### (u,v) assumed in [-1,1]**2
    def geom_transform(self, act):

        # # Initial actions
        # u = act[0]
        # v = act[1]

        # # Scale to [0,1]
        # u = (u+1.0)/2.0
        # v = (v+1.0)/2.0

        # # Radius and angle
        # r = self.r_min + u*(self.r_max - self.r_min)
        # t = self.t_min + v*(self.t_max - self.t_min)

        # # Final position
        # x = r*math.cos(t)
        # y = r*math.sin(t)

        # return np.array([x, y])

        # Initial actions
        u = act[0]
        v = act[1]

        # Map (u,v) in [-1,1]x[-1,1] to (uc,vc) in disk of radius 1
        if (u**2 >= v**2):
            uc = np.sign(u)*u*u/math.sqrt(u**2+v**2)
            vc = np.sign(u)*u*v/math.sqrt(u**2+v**2)
        else:
            uc = np.sign(v)*u*v/math.sqrt(u**2+v**2)
            vc = np.sign(v)*v*v/math.sqrt(u**2+v**2)

        # Convert (uc,vc) to (r,t)
        r = math.sqrt(uc**2 + vc**2)
        t = math.atan2(vc, uc)

        # Add hole of radius 0.5 in the center
        rmin = 0.5
        rmax = 1.0
        r    = rmin + r*(rmax-rmin)

        # Convert back to cartesian (uc,vc) in torus
        # comprised between r=0.5 and r=1
        uc = r*math.cos(t)
        vc = r*math.sin(t)

        # Map back to (u,v) in [-1,1]x[-1,1]/[-0.5,0.5]x[-0.5,0.5]
        if (uc**2 >= vc**2):
            u = np.sign(uc)*math.sqrt(uc**2+vc**2)
            v = np.sign(uc)*math.sqrt(uc**2+vc**2)*vc/uc
        else:
            u = np.sign(vc)*math.sqrt(uc**2+vc**2)*uc/vc
            v = np.sign(vc)*math.sqrt(uc**2+vc**2)

        # Map to (u,v) in [-2,2]x[-2,2]/[-1,1]x[-1,1]
        u = 2.0*u
        v = 2.0*v

        # Map to desired (x,y) bounds
        su = np.sign(u)
        sv = np.sign(v)
        xs = self.rx_min*su
        ys = self.ry_min*sv
        xe = self.rx_max*su
        ye = self.ry_max*sv
        #xs = (0.5*(main_size+obs_size) + margin)*su
        #xe = (x_lim - 0.5*obs_size)*su
        #ys = (0.5*(main_size+obs_size) + margin)*sv
        #ye = (y_lim - 0.5*obs_size)*sv
        x  = xs + (xe - xs)*(abs(u) - 1.0)
        y  = ys + (ye - ys)*(abs(v) - 1.0)

        return np.array([x, y])
