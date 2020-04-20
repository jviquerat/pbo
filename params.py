# Generic imports
import numpy as np

# Environment parameters
n_params     = 2
init_obs     = np.zeros(n_params)
x_min        =-5.0
x_max        = 5.0
y_min        =-5.0
y_max        = 5.0

# PBO parameters
pdf          = 'cma' # 'es'  or 'cma'
loss         = 'vpg' # 'vpg' or 'ppo'
n_gen        = 30
n_ind        = 6
n_avg        = 5
lr           = 5.0e-3
mu_arch      = [1]
sg_arch      = [1]
mu_epochs    = 64
sg_epochs    = 64
clip_grd     = 10.0
sg_batch     = 5
clip_pol     = 0.5
clip_adv     = True

