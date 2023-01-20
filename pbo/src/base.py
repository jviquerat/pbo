# Generic imports
import math
import numpy as np

# Custom imports
from pbo.src.agent import *

###############################################
### Base class for pbo
class base_pbo:
    def __init__(self):
        pass

    # Get data history
    def get_history(self, n_gen):

        # Starting and ending indices based on the required nb of generations
        start     = max(0,self.idx - n_gen*self.n_ind)
        end       = self.idx
        buff_size = end - start
        n_gen     = buff_size//self.n_ind

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Draw elements as lists
        buff_obs = [self.obs[i] for i in sample]
        buff_act = [self.act[i] for i in sample]
        buff_adv = [self.adv[i] for i in sample]

        # Reshape
        buff_obs = tf.reshape(buff_obs, [buff_size, self.obs_dim])
        buff_act = tf.reshape(buff_act, [buff_size, self.act_dim])
        buff_adv = tf.reshape(buff_adv, [buff_size])

        return buff_obs, buff_act, buff_adv, n_gen

    # Get actions from network
    def get_actions(self, state, n):
        raise NotImplementedError

    # Printings
    def print_generation(self, gen, ep, rwd):

        if (gen == self.n_gen-1): end = '\n'
        if (gen != self.n_gen-1): end = '\r'
        r = f"{rwd:.3e}"
        print('#   Generation #'+str(gen)+', best reward '+str(r)+' at ep '+str(ep)+'                 ', end=end)

    # Store transitions into buffer
    def store_transition(self, obs, act, acc, rwd, n):

        for cpu in range(n):
            self.obs[self.idx] = obs[cpu]
            self.act[self.idx] = act[cpu]
            self.acc[self.idx] = acc[cpu]
            self.rwd[self.idx] = rwd[cpu]
            self.idx          += 1

    # Store learning data
    def store_learning_data(self, gen, bst_ep, bst_rwd, bst_acc):

        self.bst_gen[gen] = gen
        self.bst_ep [gen] = bst_ep
        self.bst_rwd[gen] = bst_rwd
        self.bst_acc[gen] = bst_acc

    # Write learning data
    def write_learning_data(self, path, run):

        # Data related to current run
        filename = path+'/pbo_'+str(run)
        np.savetxt(filename,
                   np.hstack([np.reshape(self.gen,        (-1,1)),
                              np.reshape(self.ep,         (-1,1)),
                              np.reshape(self.rwd*(-1.0), (-1,1)),
                              self.acc]),
                   fmt='%.5e')

        # Data for future averaging
        filename = path+'/pbo_bst_'+str(run)
        np.savetxt(filename,
                   np.hstack([np.reshape(self.bst_gen,        (-1,1)),
                              np.reshape(self.bst_ep,         (-1,1)),
                              np.reshape(self.bst_rwd*(-1.0), (-1,1)),
                              self.bst_acc]),
                   fmt='%.5e')

    # Compute advantages
    def compute_advantages(self):

        # Decay
        self.adv[:] *= self.adv_decay

        # Start and end indices of last generation
        start   = max(0,self.idx - self.n_ind)
        end     = self.idx

        # Compute normalized advantage
        avg_rwd = np.mean(self.rwd[start:end])
        std_rwd = np.std( self.rwd[start:end])
        adv     = (self.rwd[start:end] - avg_rwd)/(std_rwd + 1.0e-12)

        # Clip advantages if required
        if (self.adv_clip):
            adv = np.maximum(adv, 0.0)

        # Store
        self.adv[start:end] = adv[:]

    # Train networks
    def train_networks(self):
        raise NotImplementedError
