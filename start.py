# Generic imports
import os
import sys
import json
import time
import collections
import numpy as np

# Custom imports
from pbo.training import *

########################
# Parameters decoder to collect json file
########################
def params_decoder(p_dict):
    return collections.namedtuple('X', p_dict.keys())(*p_dict.values())

########################
# Average training over multiple runs
########################

# Check command-line input for json file
if (len(sys.argv) == 2):
    json_file = sys.argv[1]
else:
    print('Command line error, please use as follows:')
    print('python3 start.py my_file.json')

# Read json parameter file
with open(json_file, "r") as f:
    params = json.load(f, object_hook=params_decoder)

# Storage arrays
res_path   = 'results'
n_data     = 5
gen        = np.zeros((              params.n_gen),         dtype=int)
data       = np.zeros((params.n_avg, params.n_gen, n_data), dtype=float)
avg_data   = np.zeros((              params.n_gen, n_data), dtype=float)
stdp_data  = np.zeros((              params.n_gen, n_data), dtype=float)
stdm_data  = np.zeros((              params.n_gen, n_data), dtype=float)

# Open storage repositories
if (not os.path.exists(res_path)):
    os.makedirs(res_path)

t         = time.localtime()
path_time = time.strftime("%H-%M-%S", t)
path      = res_path+'/'+params.env_name+'_'+str(path_time)
if (not os.path.exists(path)):
    os.makedirs(path)

for i in range(params.n_avg):
    print('### Avg run #'+str(i))
    start_time = time.time()
    launch_training(params, path, i)
    dt = time.time() - start_time
    print('#   Elapsed time: {:.3f} seconds'.format(dt))

    f   = np.loadtxt(path+'/pbo.dat')
    gen = f[:params.n_gen,0]
    for j in range(n_data):
        data[i,:,j] = f[:params.n_gen,j+1]

# Write to file
file_out  = path+'/pbo_avg.dat'
array     = np.vstack(gen)
for j in range(n_data):
    avg     = np.mean(data[:,:,j], axis=0)
    std     = np.std (data[:,:,j], axis=0)

    if (j == 0):
        log_avg = np.log(avg)
        log_std = 0.434*std/avg
        log_p   = log_avg + log_std
        log_m   = log_avg - log_std
        p       = np.exp(log_p)
        m       = np.exp(log_m)
    else:
        avg   = np.mean(data[:,:,j], axis=0)
        std   = np.std (data[:,:,j], axis=0)
        p     = avg + std
        m     = avg - std
    array   = np.hstack((array,np.vstack(avg)))
    array   = np.hstack((array,np.vstack(p)))
    array   = np.hstack((array,np.vstack(m)))

np.savetxt(file_out, array, fmt='%.5e')
os.system('gnuplot -c plot/plot.gnu '+path)
