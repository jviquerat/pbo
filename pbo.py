# Generic imports
import gym
import numpy as np

# Custom imports
from agent import *

###############################################
### Class pbo
### A PBO agent for optimization
class pbo:
    def __init__(self, params, act_dim):

        # Initialize from arguments
        self.pdf        = params.pdf
        self.act_dim    = act_dim
        self.obs_dim    = 1
        self.mu_dim     = act_dim
        self.sg_dim = self.act_dim
        if (self.pdf == 'cma-full'):
            self.sg_dim = int(self.act_dim*(self.act_dim + 1)/2)
        self.n_gen      = params.n_gen
        self.n_ind      = params.n_ind
        self.size       = self.n_gen*self.n_ind
        self.n_cpu      = params.n_cpu

        self.sg_batch   = params.sg_batch
        self.lr         = params.lr
        self.mu_epochs  = params.mu_epochs
        self.sg_epochs  = params.sg_epochs
        self.adv_clip   = params.adv_clip
        self.grd_clip   = params.grd_clip

        # Build mu network
        self.net_mu     = nn(params.mu_arch,
                             self.mu_dim,
                             'tanh',
                             self.grd_clip,
                             self.lr)

        # Build sigma network
        if (self.pdf in ['es', 'cma-diag']):
            self.net_sg = nn(params.sg_arch,
                             self.mu_dim,
                             'softplus',
                             self.grd_clip,
                             self.lr)
        if (self.pdf in ['cma-full']):
            self.net_sg = nn_cma(params.sg_arch,
                                 self.mu_dim,
                                 self.grd_clip,
                                 self.lr)

        # Init network parameters
        dummy = self.net_mu(tf.ones([1,self.obs_dim]))
        dummy = self.net_sg(tf.ones([1,self.obs_dim]))

        # Storing buffers
        self.idx     = 0

        self.gen     = np.zeros( self.size,                 dtype=np.int32)
        self.ep      = np.zeros( self.size,                 dtype=np.int32)
        self.obs     = np.zeros((self.size,  self.obs_dim), dtype=np.float32)
        self.act     = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.acc     = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.adv     = np.zeros( self.size,                 dtype=np.float32)
        self.rwd     = np.zeros( self.size,                 dtype=np.float32)
        self.mu      = np.zeros((self.size,  self.mu_dim),  dtype=np.float32)
        self.sg      = np.zeros((self.size,  self.sg_dim),  dtype=np.float32)

        self.bst_acc = np.zeros((self.n_gen, self.act_dim), dtype=np.float32)
        self.bst_rwd = np.zeros( self.n_gen,                dtype=np.float32)
        self.bst_gen = np.zeros( self.n_gen,                dtype=np.int32)
        self.bst_ep  = np.zeros( self.n_gen,                dtype=np.int32)

        self.ls_mu   = np.zeros( self.n_gen,                dtype=np.float32)
        self.ls_sg   = np.zeros( self.n_gen,                dtype=np.float32)
        self.nrm_mu  = np.zeros( self.n_gen,                dtype=np.float32)
        self.nrm_sg  = np.zeros( self.n_gen,                dtype=np.float32)
        self.lr_mu   = np.zeros( self.n_gen,                dtype=np.float32)
        self.lr_sg   = np.zeros( self.n_gen,                dtype=np.float32)

    # Get batch of observations, actions and rewards
    def get_batch(self, n_batch):

        # Starting and ending indices based on the required size of batch
        start = max(0,self.idx - n_batch*self.n_ind)
        end   = self.idx

        return self.obs[start:end], self.act[start:end], \
               self.adv[start:end], self.mu [start:end], \
               self.sg [start:end]

    # Get actions from network
    def get_actions(self, state, n):

        # Predict mu
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float32)
        mu = self.net_mu.call(x)
        mu = np.asarray(mu)[0]

        # Predict sigma
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float32)
        sg = self.net_sg.call(x)
        sg = np.asarray(sg)[0]

        # Define pdf
        if (self.pdf == 'es'):
            pdf = tfd.Normal(mu, sg)
        if (self.pdf == 'cma-diag'):
            pdf = tfd.MultivariateNormalDiag(mu, sg)
        if (self.pdf == 'cma-full'):
            diag = sg[0:self.act_dim]
            scl  = tf.tensordot(diag,tf.transpose(diag),axes=0)
            out  = tf.zeros([self.act_dim, self.act_dim], tf.float32)
            diag = sg[self.act_dim:]
            out  = tf.linalg.set_diag(out,diag,k=-1)
            out  = tf.linalg.set_diag(out,np.ones(self.act_dim),k=0)

            cov  = tf.math.multiply(scl, out)
            pdf  = tfd.MultivariateNormalTriL(mu, cov)

        # Draw actions
        ac = pdf.sample(n)
        ac = np.asarray(ac)

        # For convenience, mu and sg are returned
        # with the same dimension as actions
        mu = np.tile(mu,(n,1))
        sg = np.tile(sg,(n,1))

        return ac, mu, sg

    # Train networks
    def train_networks(self):

        # Get learning rates
        lr_mu = self.net_mu.opt._decayed_lr(tf.float32)
        lr_sg = self.net_sg.opt._decayed_lr(tf.float32)

        # Sigma network uses larger batch to simulate rank-mu update
        n_batch               = self.sg_batch
        obs, act, adv, mu, sg = self.get_batch(n_batch)

        # Account for insufficient batch history
        n_batch = min(n_batch,len(obs)//self.n_ind)

        # Update sigma network
        for epoch in range(self.sg_epochs):

            # Randomize batch
            bff_size = self.n_ind*n_batch
            btc_size = self.n_ind
            btc      = self.prep_data(obs, act, adv, mu, sg,
                                      bff_size, btc_size)
            btc_obs  = btc[0]
            btc_act  = btc[1]
            btc_adv  = btc[2]
            btc_mu   = btc[3]
            btc_sg   = btc[4]

            ls_sg, nrm_sg = self.train_sg(btc_obs, btc_adv, btc_act, btc_mu)

        # Mu network uses standard batch size
        obs, act, adv, mu, sg = self.get_batch(1)

        # Update mu network
        for epoch in range(self.mu_epochs):

            # Randomize batch
            bff_size = self.n_ind
            btc_size = self.n_ind
            btc      = self.prep_data(obs, act, adv, mu, sg,
                                      bff_size, btc_size)
            btc_obs  = btc[0]
            btc_act  = btc[1]
            btc_adv  = btc[2]
            btc_mu   = btc[3]
            btc_sg   = btc[4]

            ls_mu, nrm_mu = self.train_mu(btc_obs, btc_adv, btc_act, btc_sg)

        # Return infos
        return [ls_mu, ls_sg, nrm_mu, nrm_sg, lr_mu, lr_sg]

    # Printings
    def print_generation(self, gen):

        # Print
        if (gen == self.n_gen-1): end = '\n'
        if (gen != self.n_gen-1): end = '\r'
        print('#   Generation #'+str(gen), end=end)

    # Store transitions into buffer
    def store_transition(self, obs, act, acc, rwd, mu, sg, n):

        # Fill buffers
        for cpu in range(n):
            self.obs[self.idx] = obs[cpu]
            self.act[self.idx] = act[cpu]
            self.acc[self.idx] = acc[cpu]
            self.rwd[self.idx] = rwd[cpu]
            self.mu [self.idx] = mu [cpu]
            self.sg [self.idx] = sg [cpu]
            self.idx          += 1

    # Store learning data
    def store_learning_data(self, gen, ep, bst_rwd, bst_acc, data):

        # Store a few things
        self.bst_gen[gen] = gen
        self.bst_ep [gen] = ep
        self.bst_rwd[gen] = bst_rwd
        self.bst_acc[gen] = bst_acc
        self.ls_mu  [gen] = data[0]
        self.ls_sg  [gen] = data[1]
        self.nrm_mu [gen] = data[2]
        self.nrm_sg [gen] = data[3]

    # Write learning data
    def write_learning_data(self, path, run):

        # Data related to current run
        filename = path+'/pbo_'+str(run)
        np.savetxt(filename,
                   np.hstack([np.reshape(self.gen,        (-1,1)),
                              np.reshape(self.ep,         (-1,1)),
                              np.reshape(self.rwd*(-1.0), (-1,1)),
                              self.acc,
                              self.mu,
                              self.sg]),
                   fmt='%.5e')

        # Data for future averaging
        filename = path+'/pbo.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.bst_gen+1,      (-1,1)),
                              np.reshape(self.bst_rwd*(-1.0), (-1,1)),
                              np.reshape(self.ls_mu,          (-1,1)),
                              np.reshape(self.ls_sg,          (-1,1)),
                              np.reshape(self.nrm_mu,         (-1,1)),
                              np.reshape(self.nrm_sg,         (-1,1)),
                              np.reshape(self.lr_mu,          (-1,1)),
                              np.reshape(self.lr_sg,          (-1,1)),
                              self.bst_acc]),
                   fmt='%.5e')

    # Compute advantages
    def compute_advantages(self):

        # Start and end indices of last generation
        start   = max(0,self.idx - self.n_ind)
        end     = self.idx

        # Compute normalized advantage
        avg_rwd = np.mean(self.rwd[start:end])
        std_rwd = np.std( self.rwd[start:end])
        adv     = (self.rwd[start:end] - avg_rwd)/(std_rwd + 1.0e-7)

        # Clip advantages if required
        if (self.adv_clip):
            adv = np.maximum(adv, 0.0)

        # Store
        self.adv[start:end] = adv

    # Prepare data for training
    def prep_data(self, obs, act, adv, mu, sg, bff_size, btc_size):

        # Randomize batch
        sample = np.arange(bff_size)
        np.random.shuffle(sample)
        sample = sample[:btc_size]

        # Draw elements as lists
        btc_obs = [obs[i] for i in sample]
        btc_act = [act[i] for i in sample]
        btc_adv = [adv[i] for i in sample]
        btc_mu  = [mu [i] for i in sample]
        btc_sg  = [sg [i] for i in sample]

        # Reshape
        btc_obs = tf.reshape(tf.cast(btc_obs, tf.float32),
                             [btc_size, self.obs_dim])
        btc_act = tf.reshape(tf.cast(btc_act, tf.float32),
                             [btc_size, self.act_dim])
        btc_adv = tf.reshape(tf.cast(btc_adv, tf.float32),
                             [btc_size])
        btc_mu  = tf.reshape(tf.cast(btc_mu,  tf.float32),
                             [btc_size, self.mu_dim])
        btc_sg  = tf.reshape(tf.cast(btc_sg,  tf.float32),
                             [btc_size, self.sg_dim])

        return btc_obs, btc_act, btc_adv, btc_mu, btc_sg

    # Train sg network
    @tf.function
    def train_sg(self, obs, adv, act, mu):
        var = self.net_sg.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            sg = tf.convert_to_tensor(self.net_sg.call(obs))

            # Compute loss
            loss = self.get_loss(obs, adv, act, mu, sg)

        # Apply gradients
        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_sg.opt.apply_gradients(zip(grads, var))

        return loss, norm

    # Train mu network
    @tf.function
    def train_mu(self, obs, adv, act, sg):
        var = self.net_mu.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            mu = tf.convert_to_tensor(self.net_mu.call(obs))

            # Compute loss
            loss = self.get_loss(obs, adv, act, mu, sg)

        # Apply gradients
        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_mu.opt.apply_gradients(zip(grads, var))

        return loss, norm

    # Compute loss
    @tf.function
    def get_loss(self, obs, adv, act, mu, sg):

        # Compute pdf
        if (self.pdf == 'es'):
            pdf = tfd.Normal(mu[0], sg[0])
            log = tf.reduce_sum(pdf.log_prob(act), axis=1)
        if (self.pdf == 'cma-diag'):
            pdf = tfd.MultivariateNormalDiag(mu[0], sg[0])
            log = pdf.log_prob(act)
        if (self.pdf == 'cma-full'):
            diag = sg[0,0:self.act_dim]
            scl  = tf.tensordot(diag,tf.transpose(diag),axes=0)
            out  = tf.zeros([self.act_dim, self.act_dim], tf.float32)
            diag = sg[0,self.act_dim:]
            out  = tf.linalg.set_diag(out,diag,k=-1)
            out  = tf.linalg.set_diag(out,np.ones(self.act_dim),k=0)

            cov  = tf.math.multiply(scl, out)
            pdf  = tfd.MultivariateNormalTriL(mu[0], cov)
            log  = pdf.log_prob(act)

        # Compute loss
        s    = tf.multiply(adv, log)
        loss =-tf.reduce_mean(s)

        return loss
