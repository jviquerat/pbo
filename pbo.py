# Generic imports
import os
import gym
import warnings
import numpy as np

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow             as tf
import tensorflow.keras       as tk
import tensorflow_probability as tfp
import tensorflow_addons      as tfa
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal

# Define alias
tfd = tfp.distributions

# Custom imports
from environment import *

###############################################
### Generic neural network for mu and sigma predictions
class nn(Model):
    def __init__(self, arch, dim, last, clip_grd, lr):
        super(nn, self).__init__()

        # Initialize network as empty list
        self.net = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer=Orthogonal(gain=0.5),
                                  activation = 'sigmoid'))

        # Define last layer
        self.net.append(Dense(dim,
                              kernel_initializer=Orthogonal(gain=0.1),
                              activation = last))

        # Define optimizer
        self.opt = tfa.optimizers.RectifiedAdam(lr       = lr,
                                                clipnorm = clip_grd)
        
    # Network forward pass
    @tf.function
    def call(self, x):

        # Compute output
        for layer in range(len(self.net)):
            x = self.net[layer](x)

        return x

###############################################
### Class pbo
### A PBO agent for optimization
class pbo:
    def __init__(self,
                 pdf, loss, act_dim, obs_dim, n_gen, n_ind, lr,
                 mu_epochs, sg_epochs, clip_grd, sg_batch, clip_adv,
                 clip_pol, mu_arch, sg_arch):

        # Initialize from arguments
        self.pdf        = pdf
        self.loss       = loss
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.mu_dim     = act_dim
        if (pdf == 'cma-full'):
            self.sg_dim = int(act_dim*(act_dim + 1)/2)
        else:
            self.sg_dim = act_dim
        self.n_gen      = n_gen
        self.n_ind      = n_ind
        self.size       = self.n_gen*self.n_ind

        self.sg_batch   = sg_batch
        self.lr         = lr
        self.mu_epochs  = mu_epochs
        self.sg_epochs  = sg_epochs
        self.clip_adv   = clip_adv
        self.clip_pol   = clip_pol
        self.clip_grd   = clip_grd

        # Build networks
        self.net_mu = nn(mu_arch, self.mu_dim, 'linear',   clip_grd, lr)
        self.net_sg = nn(sg_arch, self.sg_dim, 'softplus', clip_grd, lr)

        # Init network parameters
        dummy = self.net_mu(tf.ones([1,self.obs_dim]))
        dummy = self.net_sg(tf.ones([1,self.obs_dim+self.mu_dim]))

        # If loss is ppo-style
        if (self.loss == 'ppo'):
            self.net_mu_old = nn(mu_arch,self.mu_dim,'linear',  clip_grd,lr)
            self.net_sg_old = nn(sg_arch,self.sg_dim,'softplus',clip_grd,lr)
            dummy      = self.net_mu_old(tf.ones([1,self.obs_dim]))
            dummy      = self.net_sg_old(tf.ones([1,self.obs_dim]))
            mu_weights = self.net_mu.get_weights()
            sg_weights = self.net_sg.get_weights()
            self.net_mu_old.set_weights(mu_weights)
            self.net_sg_old.set_weights(sg_weights)

        # Storing buffers
        self.idx     = 0

        self.gen     = np.zeros( self.size,                 dtype=np.int32)
        self.ep      = np.zeros( self.size,                 dtype=np.int32)
        self.obs     = np.zeros((self.size,  self.obs_dim), dtype=np.float32)
        self.act     = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.cac     = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.rwd     = np.zeros( self.size,                 dtype=np.float32)
        self.mu      = np.zeros((self.size,  self.mu_dim),  dtype=np.float32)
        self.sg      = np.zeros((self.size,  self.sg_dim),  dtype=np.float32)

        self.bst_cac = np.zeros((self.n_gen, self.act_dim), dtype=np.float32)
        self.bst_rwd = np.zeros( self.n_gen,                dtype=np.float32)
        self.bst_gen = np.zeros( self.n_gen,                dtype=np.int32)
        self.bst_ep  = np.zeros( self.n_gen,                dtype=np.int32)

        self.adv     = np.zeros( self.size,                 dtype=np.float32)

    # Get batch of observations, actions and rewards
    def get_batch(self, n_batch):

        # Starting and ending indices based on the required size of batch
        start = max(0,self.idx - n_batch*self.n_ind)
        end   = self.idx

        return self.obs[start:end], self.act[start:end], \
               self.adv[start:end], self.mu [start:end], \
               self.sg [start:end]

    # Get actions from network
    def get_actions(self, state):

        # Predict mu
        x  = tf.convert_to_tensor([state], dtype=tf.float32)
        mu = self.net_mu.call(x)
        mu = np.asarray(mu)[0]

        # Predict sigma
        x  = np.hstack((state,mu))
        x  = tf.convert_to_tensor([x], dtype=tf.float32)
        sg = self.net_sg.call(x)
        sg = np.asarray(sg)[0]

        # Empty actions array
        actions = np.zeros(self.act_dim)

        # Define pdf
        if (self.pdf == 'es'):
            pdf = tfd.Normal(mu, sg)
        if (self.pdf == 'cma-diag'):
            pdf = tfd.MultivariateNormalDiag(mu, sg)
        if (self.pdf == 'cma-full'):
            cov = tfp.math.fill_triangular(sg)
            pdf = tfd.MultivariateNormalTriL(mu, cov)

        # Draw actions
        actions = pdf.sample(1)
        actions = np.asarray(actions)[0]

        return actions, mu, sg

    # Train networks
    def train_networks(self):

        # Sigma network uses larger batch to simulate rank-mu update
        n_batch               = self.sg_batch
        obs, act, adv, mu, sg = self.get_batch(n_batch)

        # Account for insufficient batch history
        n_batch = min(n_batch,len(obs)//self.n_ind)

        # Update sigma network
        for epoch in range(self.sg_epochs):

            # Randomize batch
            sample = np.arange(self.n_ind*n_batch)
            np.random.shuffle(sample)
            sample = sample[:self.n_ind]

            btc_obs = [obs[i] for i in sample]
            btc_adv = [adv[i] for i in sample]
            btc_mu  = [mu [i] for i in sample]
            btc_act = [act[i] for i in sample]

            btc_obs = tf.reshape(tf.cast(btc_obs, tf.float32),
                                 [self.n_ind, self.obs_dim])
            btc_adv = tf.reshape(tf.cast(btc_adv, tf.float32),
                                 [self.n_ind])
            btc_mu  = tf.reshape(tf.cast(btc_mu,  tf.float32),
                                 [self.n_ind, self.mu_dim])
            btc_act = tf.reshape(tf.cast(btc_act, tf.float32),
                                 [self.n_ind, self.act_dim])

            self.train_sg(btc_obs, btc_adv, btc_act, btc_mu)

        # Mu network uses standard batch size
        obs, act, adv, mu, sg = self.get_batch(1)

        # Update mu network
        for epoch in range(self.mu_epochs):

            # Randomize batch
            sample = np.arange(self.n_ind)
            np.random.shuffle(sample)

            btc_obs = [obs[i] for i in sample]
            btc_adv = [adv[i] for i in sample]
            btc_sg  = [sg [i] for i in sample]
            btc_act = [act[i] for i in sample]

            btc_obs = tf.reshape(tf.cast(btc_obs, tf.float32),
                                 [self.n_ind, self.obs_dim])
            btc_adv = tf.reshape(tf.cast(btc_adv, tf.float32),
                                 [self.n_ind])
            btc_sg  = tf.reshape(tf.cast(btc_sg,  tf.float32),
                                 [self.n_ind, self.sg_dim])
            btc_act = tf.reshape(tf.cast(btc_act, tf.float32),
                                 [self.n_ind, self.act_dim])

            self.train_mu(btc_obs, btc_adv, btc_act, btc_sg)

        # Update old network if loss is ppo-style
        if (self.loss == 'ppo'):
            mu_weights = self.net_mu.get_weights()
            sg_weights = self.net_sg.get_weights()
            self.net_mu_old.set_weights(mu_weights)
            self.net_sg_old.set_weights(sg_weights)

    # Store transitions into buffer
    def store_transition(self, obs, act, cac, rwd, mu, sg):

        # Fill buffers
        self.obs[self.idx] = obs
        self.act[self.idx] = act
        self.cac[self.idx] = cac
        self.rwd[self.idx] = rwd
        self.mu [self.idx] = mu
        self.sg [self.idx] = sg

        # Update index
        self.idx          += 1

    # Compute advantages
    def compute_advantages(self):

        # Start and end indices of last generation
        start        = max(0,self.idx - self.n_ind)
        end          = self.idx

        # Compute normalized advantage
        avg_rwd      = np.mean(self.rwd[start:end])
        std_rwd      = np.std( self.rwd[start:end])
        adv          = (self.rwd[start:end] - avg_rwd)/(std_rwd + 1.0e-7)

        # Clip advantages if required
        if (self.clip_adv):
            adv = np.maximum(adv,0.0)

        # Store
        self.adv[start:end] = adv

    # Train sg network
    @tf.function
    def train_sg(self, obs, adv, act, mu):
        var = self.net_sg.trainable_variables
        with tf.GradientTape() as tape:
            # Watch network variables
            tape.watch(var)

            # Network forward pass
            x  = tf.concat([obs,mu], axis=1)
            sg = tf.convert_to_tensor(self.net_sg.call(x))

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

        # PPO-style loss
        if (self.loss == 'ppo'):
            old_sg = tf.convert_to_tensor(self.net_sg_old(obs))
            old_mu = tf.convert_to_tensor(self.net_mu_old(obs))

            # Compute pdf
            if (self.pdf == 'es'):
                pdf     = tfd.Normal(    mu,     sg)
                old_pdf = tfd.Normal(old_mu, old_sg)
                log     = tf.reduce_sum(    pdf.log_prob(act), axis=1)
                old_log = tf.reduce_sum(old_pdf.log_prob(act), axis=1)
            if (self.pdf == 'cma-diag'):
                pdf     = tfd.MultivariateNormalDiag(    mu,     sg)
                old_pdf = tfd.MultivariateNormalDiag(old_mu, old_sg)
                log     =     pdf.log_prob(act)
                old_log = old_pdf.log_prob(act)
            if (self.pdf == 'cma-full'):
                cov     = tfp.math.fill_triangular(sg)
                old_cov = tfp.math.fill_triangular(old_sg)
                pdf     = tfd.MultivariateNormalTriL(    mu,     cov)
                old_pdf = tfd.MultivariateNormalTriL(old_mu, old_cov)
                log     =     pdf.log_prob(act)
                old_log = old_pdf.log_prob(act)

            # Compute surrogate
            ratio = tf.exp(log - old_log)
            s1    = tf.multiply(adv,ratio)
            clp   = tf.clip_by_value(ratio,
                                     1.0-self.clip_pol,
                                     1.0+self.clip_pol)
            s2    = tf.multiply(adv,clp)
            loss  =-tf.reduce_mean(tf.minimum(s1,s2))

        # VPG-style loss
        if (self.loss == 'vpg'):
            # Compute pdf
            if (self.pdf == 'es'):
                pdf = tfd.Normal(mu, sg)
                log = tf.reduce_sum(pdf.log_prob(act), axis=1)
            if (self.pdf == 'cma-diag'):
                pdf = tfd.MultivariateNormalDiag(mu, sg)
                log = pdf.log_prob(act)
            if (self.pdf == 'cma-full'):
                cov = tfp.math.fill_triangular(sg)
                pdf = tfd.MultivariateNormalTriL(mu, cov)
                log = pdf.log_prob(act)

            loss =-tf.reduce_mean(adv*log)

        return loss
