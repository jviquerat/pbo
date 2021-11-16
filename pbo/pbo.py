# Generic imports
import math
import numpy as np

# Custom imports
from pbo.agent import *

###############################################
### Class pbo
### A PBO agent for optimization
class pbo:
    def __init__(self, params, act_dim, obs_dim):

        # Initialize from arguments
        self.pdf        = params.pdf
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.mu_dim     = act_dim
        self.sg_dim     = act_dim
        self.cr_dim     = int(self.act_dim*(self.act_dim - 1)/2)
        self.n_gen      = params.n_gen
        self.n_ind      = params.n_ind
        self.size       = self.n_gen*self.n_ind
        self.n_cpu      = params.n_cpu

        self.mu_gen     = params.mu_gen
        self.sg_gen     = params.sg_gen
        self.cr_gen     = params.cr_gen
        self.mu_batch   = params.mu_batch
        self.sg_batch   = params.sg_batch
        self.cr_batch   = params.cr_batch
        self.lr_mu      = params.lr_mu
        self.lr_sg      = params.lr_sg
        self.lr_cr      = params.lr_cr
        self.mu_epochs  = params.mu_epochs
        self.sg_epochs  = params.sg_epochs
        self.cr_epochs  = params.cr_epochs
        self.adv_clip   = params.adv_clip
        self.adv_decay  = params.adv_decay

        # Build mu network
        self.net_mu     = nn(params.mu_arch,
                             self.mu_dim,
                             'tanh',
                             'tanh',
                             self.lr_mu)
        self.net_sg     = nn(params.sg_arch,
                             self.sg_dim,
                             'sigmoid',
                             'sigmoid',
                             self.lr_sg)
        self.net_cr     = nn(params.cr_arch,
                             self.cr_dim,
                             'sigmoid',
                             'sigmoid',
                             self.lr_cr)

        # Init network parameters
        dummy = self.net_mu(tf.ones([1,self.obs_dim]))
        dummy = self.net_sg(tf.ones([1,self.obs_dim]))
        dummy = self.net_cr(tf.ones([1,self.obs_dim]))

        # Storing buffers
        self.idx     = 0

        self.gen     = np.zeros( self.size,               dtype=np.int32)
        self.ep      = np.zeros( self.size,               dtype=np.int32)
        self.obs     = np.zeros((self.size, self.obs_dim),dtype=np.float64)
        self.act     = np.zeros((self.size, self.act_dim),dtype=np.float64)
        self.acc     = np.zeros((self.size, self.act_dim),dtype=np.float64)
        self.adv     = np.zeros( self.size,               dtype=np.float64)
        self.rwd     = np.zeros( self.size,               dtype=np.float64)
        self.mu      = np.zeros((self.size, self.mu_dim), dtype=np.float64)
        self.sg      = np.zeros((self.size, self.sg_dim), dtype=np.float64)
        self.cr      = np.zeros((self.size, self.cr_dim), dtype=np.float64)

        self.bst_acc = np.zeros((self.n_gen,self.act_dim),dtype=np.float64)
        self.bst_rwd = np.zeros( self.n_gen,              dtype=np.float64)
        self.bst_gen = np.zeros( self.n_gen,              dtype=np.int32)
        self.bst_ep  = np.zeros( self.n_gen,              dtype=np.int32)

        self.ls_mu   = np.zeros( self.n_gen,              dtype=np.float64)
        self.ls_sg   = np.zeros( self.n_gen,              dtype=np.float64)
        self.ls_cr   = np.zeros( self.n_gen,              dtype=np.float64)
        self.nrm_mu  = np.zeros( self.n_gen,              dtype=np.float64)
        self.nrm_sg  = np.zeros( self.n_gen,              dtype=np.float64)
        self.nrm_cr  = np.zeros( self.n_gen,              dtype=np.float64)
        self.lr_mu   = np.zeros( self.n_gen,              dtype=np.float64)
        self.lr_sg   = np.zeros( self.n_gen,              dtype=np.float64)
        self.lr_cr   = np.zeros( self.n_gen,              dtype=np.float64)

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

        # Predict mu
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        mu = self.net_mu.call(x)
        mu = np.asarray(mu)[0]

        # Predict sigma
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        sg = self.net_sg.call(x)
        sg = np.asarray(sg)[0]

        # Predict correlations
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        cr = self.net_cr.call(x)
        cr = np.asarray(cr)[0]

        # Define pdf
        if (self.pdf == 'es'):
            pdf = tfd.Normal(mu, sg)
        if (self.pdf == 'cma-diag'):
            pdf = tfd.MultivariateNormalDiag(mu, sg)
        if (self.pdf == 'cma-full'):
            pdf = self.get_cov_pdf(mu, sg, cr)

        # Draw actions
        ac = pdf.sample(n)
        ac = np.asarray(ac)
        ac = np.clip(ac, -1.0, 1.0)

        # For convenience, mu and sg are returned
        # with the same dimension as actions
        mu = np.tile(mu,(n,1))
        sg = np.tile(sg,(n,1))
        cr = np.tile(cr,(n,1))

        return ac, mu, sg, cr

    # Printings
    def print_generation(self, gen, rwd):

        # Print
        if (gen == self.n_gen-1): end = '\n'
        if (gen != self.n_gen-1): end = '\r'
        print('#   Generation #'+str(gen)+', best reward '+str(rwd)+'                 ', end=end)

    # Store transitions into buffer
    def store_transition(self, obs, act, acc, rwd, mu, sg, cr, n):

        # Fill buffers
        for cpu in range(n):
            self.obs[self.idx] = obs[cpu]
            self.act[self.idx] = act[cpu]
            self.acc[self.idx] = acc[cpu]
            self.rwd[self.idx] = rwd[cpu]
            self.mu [self.idx] = mu [cpu]
            self.sg [self.idx] = sg [cpu]
            self.cr [self.idx] = cr [cpu]
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
        self.ls_cr  [gen] = data[2]
        self.nrm_mu [gen] = data[3]
        self.nrm_sg [gen] = data[4]
        self.nrm_cr [gen] = data[5]

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
                              self.sg,
                              self.cr]),
                   fmt='%.5e')

        # Data for future averaging
        filename = path+'/pbo.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.bst_gen+1,      (-1,1)),
                              np.reshape(self.bst_rwd*(-1.0), (-1,1)),
                              np.reshape(self.ls_mu,          (-1,1)),
                              np.reshape(self.ls_sg,          (-1,1)),
                              np.reshape(self.ls_cr,          (-1,1)),
                              np.reshape(self.nrm_mu,         (-1,1)),
                              np.reshape(self.nrm_sg,         (-1,1)),
                              np.reshape(self.nrm_cr,         (-1,1)),
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
        self.adv[start:end] = adv

    # Train networks
    def train_networks(self):

        # Train
        ls_sg, nrm_sg = self.train_loop_sg()
        ls_cr, nrm_cr = self.train_loop_cr()
        ls_mu, nrm_mu = self.train_loop_mu()

        # Return infos
        return [ ls_mu,  ls_sg,  ls_cr,
                nrm_mu, nrm_sg, nrm_cr]

    # Train loop for mu
    def train_loop_mu(self):

        # Update sigma network
        for epoch in range(self.mu_epochs):

            obs, act, adv, n_gen = self.get_history(self.mu_gen)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.mu_batch*self.n_ind
                end      = min((btc+1)*self.mu_batch*self.n_ind,len(obs))
                btc     += 1
                if (end == len(obs)): done = True
                btc_obs  = obs[start:end]
                btc_act  = act[start:end]
                btc_adv  = adv[start:end]

                ls_mu, nrm_mu = self.train_mu(btc_obs, btc_adv, btc_act)

        return ls_mu, nrm_mu

    # Train loop for sg
    def train_loop_sg(self):

        # Update sigma network
        for epoch in range(self.sg_epochs):

            obs, act, adv, n_gen = self.get_history(self.sg_gen)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.sg_batch*self.n_ind
                end      = min((btc+1)*self.sg_batch*self.n_ind,len(obs))
                btc     += 1
                if (end == len(obs)): done = True

                btc_obs  = obs[start:end]
                btc_act  = act[start:end]
                btc_adv  = adv[start:end]

                ls_sg, nrm_sg = self.train_sg(btc_obs, btc_adv, btc_act)

        return ls_sg, nrm_sg

    # Train loop for cr
    def train_loop_cr(self):

        # If correlations are not used
        if (self.pdf != 'cma-full'): return 1.0, 1.0

        # Update sigma network
        for epoch in range(self.cr_epochs):

            obs, act, adv, n_gen = self.get_history(self.cr_gen)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.cr_batch*self.n_ind
                end      = min((btc+1)*self.cr_batch*self.n_ind,len(obs))
                btc     += 1
                if (end == len(obs)): done = True
                btc_obs  = obs[start:end]
                btc_act  = act[start:end]
                btc_adv  = adv[start:end]

                ls_cr, nrm_cr = self.train_cr(btc_obs, btc_adv, btc_act)

        return ls_cr, nrm_cr

    # Train mu network
    @tf.function
    def train_mu(self, obs, adv, act):
        var = self.net_mu.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            cr = tf.convert_to_tensor(self.net_cr.call(obs), tf.float64)
            sg = tf.convert_to_tensor(self.net_sg.call(obs), tf.float64)
            mu = tf.convert_to_tensor(self.net_mu.call(obs), tf.float64)

            # Compute loss
            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        # Apply gradients
        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_mu.opt.apply_gradients(zip(grads, var))

        return loss, norm

    # Train sg network
    @tf.function
    def train_sg(self, obs, adv, act):
        var = self.net_sg.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            cr = tf.convert_to_tensor(self.net_cr.call(obs), tf.float64)
            sg = tf.convert_to_tensor(self.net_sg.call(obs), tf.float64)
            mu = tf.convert_to_tensor(self.net_mu.call(obs), tf.float64)

            # Compute loss
            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        # Apply gradients
        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_sg.opt.apply_gradients(zip(grads, var))

        return loss, norm

    # Train cr network
    @tf.function
    def train_cr(self, obs, adv, act):
        var = self.net_cr.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            cr = tf.convert_to_tensor(self.net_cr.call(obs), tf.float64)
            sg = tf.convert_to_tensor(self.net_sg.call(obs), tf.float64)
            mu = tf.convert_to_tensor(self.net_mu.call(obs), tf.float64)

            # Compute loss
            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        # Apply gradients
        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_cr.opt.apply_gradients(zip(grads, var))

        return loss, norm

    # Compute loss
    def get_loss(self, obs, adv, act, mu, sg, cr):

        # Compute pdf
        if (self.pdf == 'es'):
            pdf = tfd.Normal(mu[0], sg[0])
            log = tf.reduce_sum(pdf.log_prob(act), axis=1)
        if (self.pdf == 'cma-diag'):
            pdf = tfd.MultivariateNormalDiag(mu[0], sg[0])
            log = pdf.log_prob(act)
        if (self.pdf == 'cma-full'):
            pdf = self.get_cov_pdf(mu[0], sg[0], cr[0])
            log = pdf.log_prob(act)

        # Compute loss
        s     = tf.multiply(adv, log)
        loss  =-tf.reduce_mean(s)

        return loss

    # Compute full cov pdf
    def get_cov_pdf(self, mu, sg, cr):
        cov  = self.get_cov(sg, cr)
        scl  = tf.linalg.cholesky(cov)
        pdf  = tfd.MultivariateNormalTriL(mu, scl)

        return pdf

    # Compute covariance matrix
    def get_cov(self, sg, cr):

        ### Use correlative angle matrix ###
        # Extract sigmas and thetas
        sigmas = 0.85*sg
        thetas = cr*math.pi

        # Build initial theta matrix
        t   = tf.ones([self.act_dim,self.act_dim], tf.float64)*math.pi/2.0
        t   = tf.linalg.set_diag(t, tf.zeros(self.act_dim, tf.float64), k=0)
        idx = 0
        for dg in range(self.act_dim-1):
            diag = tf.cast(thetas[idx:idx+self.act_dim-(dg+1)], tf.float64)
            idx += self.act_dim-(dg+1)
            t    = tf.linalg.set_diag(t, diag, k=-(dg+1))
        cor = tf.cos(t)

        # Correct upper part to exact zero
        for dg in range(self.act_dim-1):
            size = self.act_dim-(dg+1)
            cor  = tf.linalg.set_diag(cor, tf.zeros(size, tf.float64), k=(dg+1))

        # Roll and compute additional terms
        for roll in range(self.act_dim-1):
            vec = tf.ones([self.act_dim, 1], tf.float64)
            vec = tf.scalar_mul(math.pi/2, vec)
            t   = tf.concat([vec, t[:, :self.act_dim-1]], axis=1)
            for dg in range(self.act_dim-1):
                zero = tf.zeros(self.act_dim-(dg+1), tf.float64)
                t    = tf.linalg.set_diag(t, zero, k=dg+1)
            cor = tf.multiply(cor, tf.sin(t))

        cor = tf.matmul(cor, tf.transpose(cor))
        scl = tf.zeros([self.act_dim, self.act_dim], tf.float64)
        scl = tf.linalg.set_diag(scl, sigmas, k=0)
        cov = tf.matmul(scl, cor)
        cov = tf.matmul(cov, scl)

        return cov
