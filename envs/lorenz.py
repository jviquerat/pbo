# Generic imports
import os
import math
import numpy           as     np
from   math            import tanh, sqrt
from   scipy.integrate import odeint

# Custom imports
from   envs.base_env   import base_env

###############################################
### Base environment for Lorenz control
class lorenz(base_env):

    # Create object
    def __init__(self, path):

        # Problem parameters
        self.t_max     = 25.0
        self.x0        = 10.0
        self.y0        = 10.0
        self.z0        = 10.0
        self.init_time = 5.0
        self.t0        =-self.init_time
        self.u0        = 0.0
        self.dt        = 0.01
        self.steps     = math.floor((self.t_max-self.t0)/self.dt)

        # Environment parameters
        self.act_size = 4
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.ones(self.act_size)*(-1.0)
        self.x_max    = np.ones(self.act_size)*( 1.0)
        self.x_0      = np.ones(self.act_size)*( 0.0)

        # Set path
        self.path      = path
        os.makedirs(path, exist_ok=True)

    # Reset variables
    def reset(self):

        # Init values and lists
        self.t     = self.t0
        self.x     = self.x0
        self.y     = self.y0
        self.z     = self.z0
        self.u     = self.u0

        self.lst_t  = []
        self.lst_x  = []
        self.lst_y  = []
        self.lst_z  = []
        self.lst_dx = []
        self.lst_dy = []
        self.lst_dz = []
        self.lst_u  = []

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

    # Lorenz system
    def lorenz(self, x, y, z):

        sigma   = 10.0
        rho     = 28.0
        beta    = 8.0/3.0

        dxdt    = sigma*(y - x)
        dydt    = x*(rho - z) - y
        dzdt    = x*y - beta*z

        return dxdt, dydt, dzdt

    # Call function for odeint
    def ode_lorenz(self, xv, t, act):

        x, y, z          = xv
        dxdt, dydt, dzdt = self.lorenz(x, y, z)
        u                = self.ctrl(act, dxdt, dydt, dzdt, t)
        dydt            += u

        return dxdt, dydt, dzdt

    # Integrate system
    def integrate(self, act):

        # Integrate
        t = np.linspace(self.t0, self.t_max, self.steps)
        f = odeint(self.ode_lorenz, (self.x, self.y, self.z), t,
                   args=(act,))

        # Store
        self.lst_t = t
        self.lst_x = f.T[0,:]
        self.lst_y = f.T[1,:]
        self.lst_z = f.T[2,:]

        # Rebuild control from history
        for step in range(len(self.lst_x)):
            x = self.lst_x[step]
            y = self.lst_y[step]
            z = self.lst_z[step]
            t = self.lst_t[step]

            dxdt, dydt, dzdt = self.lorenz(x,y,z)
            self.lst_dx.append(dxdt)
            self.lst_dy.append(dydt)
            self.lst_dz.append(dzdt)

            u = self.ctrl(act, dxdt, dydt, dzdt, t)
            self.lst_u.append(u)

    # Control law
    def ctrl(self, act, dxdt, dydt, dzdt, t):

        if (t >= 0.0):
            u = tanh(act[0]*dxdt/10.0
                   + act[1]*dydt/20.0
                   + act[2]*dzdt/40.0
                   + act[3])
        else:
            u = 0.0

        return u

    # Pure virtual
    def get_rwd(self):
        raise NotImplementedError()

    # Dump
    def dump(self, ep):

        filename = self.path+'/'+str(ep)+'.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.lst_t,  (-1,1)),
                              np.reshape(self.lst_x,  (-1,1)),
                              np.reshape(self.lst_y,  (-1,1)),
                              np.reshape(self.lst_z,  (-1,1)),
                              np.reshape(self.lst_dx, (-1,1)),
                              np.reshape(self.lst_dy, (-1,1)),
                              np.reshape(self.lst_dz, (-1,1)),
                              np.reshape(self.lst_u,  (-1,1))]),
                   fmt='%.5e')
