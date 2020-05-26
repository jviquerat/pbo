# Generic imports
import numpy as np

# Environment parameters
n_params     = 2
init_obs     = np.zeros(n_params)
# x_min        =-5.0
# x_max        = 5.0
# y_min        =-5.0
# y_max        = 5.0

x_min        =-2.0
x_max        = 2.0
y_min        =-1.0
y_max        = 3.0

# PBO parameters
pdf          = 'cma-full' # 'es', 'cma-diag' or 'cma-full'
loss         = 'vpg'      # 'vpg' or 'ppo'
n_gen        = 150         # nb of generations
n_ind        = 6          # nb of individuals per generation
n_avg        = 5          # nb of runs to average results
lr           = 5.0e-3     # learning rate
mu_arch      = [4,4]        # architecture for mu    network
sg_arch      = [4,4]        # architecture for sigma network
mu_epochs    = 64         # nb of epochs for mu    update
sg_epochs    = 64        # nb of epochs for sigma update
clip_grd     = 0.5       # value for gradient clipping
sg_batch     = 10          # batch size for sigma update (>1 is off-policy)
clip_pol     = 0.5        # value for policy   clipping (ppo only)
clip_adv     = True       # True to clip negative advantages

