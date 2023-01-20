# Custom imports
from pbo.src.base import *

###############################################
### Class discrete
### A PBO agent for discrete optimization
class discrete(base_pbo):
    def __init__(self, params, act_dim, obs_dim):

        # Initialize from arguments
        self.pdf        = params.pdf
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        #self.mu_dim     = act_dim
        #self.sg_dim     = act_dim
        #self.cr_dim     = int(self.act_dim*(self.act_dim - 1)/2)
        self.n_gen      = params.n_gen
        self.n_ind      = params.n_ind
        self.size       = self.n_gen*self.n_ind
        self.n_cpu      = params.n_cpu

        # Handle act_dim=1
        if (self.act_dim==1): self.pdf = 'es'

        #self.mu_gen     = params.mu_gen
        #self.sg_gen     = params.sg_gen
        #self.cr_gen     = params.cr_gen
        #self.mu_batch   = params.mu_batch
        #self.sg_batch   = params.sg_batch
        #self.cr_batch   = params.cr_batch
        #self.lr_mu      = params.lr_mu
        #self.lr_sg      = params.lr_sg
        #self.lr_cr      = params.lr_cr
        #self.mu_epochs  = params.mu_epochs
        #self.sg_epochs  = params.sg_epochs
        #self.cr_epochs  = params.cr_epochs
        self.adv_clip   = params.adv_clip
        self.adv_decay  = params.adv_decay

        # Build mu network
        #self.net_mu     = nn(params.mu_arch,
        #                     self.mu_dim,
        #                     'relu',
        #                     'tanh',
        #                     self.lr_mu)
        #self.net_sg     = nn(params.sg_arch,
        #                     self.sg_dim,
        #                     'tanh',
        #                     'sigmoid',
        #                     self.lr_sg)
        #self.net_cr     = nn(params.cr_arch,
        #                     self.cr_dim,
        #                     'tanh',
        #                     'sigmoid',
        #                     self.lr_cr)

        # Init network parameters
        #dummy = self.net_mu(tf.ones([1,self.obs_dim]))
        #dummy = self.net_sg(tf.ones([1,self.obs_dim]))
        #dummy = self.net_cr(tf.ones([1,self.obs_dim]))

        # Storing buffers
        self.idx     = 0

        self.gen     = np.zeros( self.size,               dtype=np.int32)
        self.ep      = np.zeros( self.size,               dtype=np.int32)
        self.obs     = np.zeros((self.size, self.obs_dim),dtype=np.float64)
        self.act     = np.zeros((self.size, self.act_dim),dtype=np.float64)
        self.acc     = np.zeros((self.size, self.act_dim),dtype=np.float64)
        self.adv     = np.zeros( self.size,               dtype=np.float64)
        self.rwd     = np.zeros( self.size,               dtype=np.float64)

        self.bst_gen = np.zeros( self.n_gen,              dtype=np.int32)
        self.bst_ep  = np.zeros( self.n_gen,              dtype=np.int32)
        self.bst_acc = np.zeros((self.n_gen,self.act_dim),dtype=np.float64)
        self.bst_rwd = np.zeros( self.n_gen,              dtype=np.float64)

    # Get actions from network
    def get_actions(self, state, n):

        # # Predict mu
        # x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        # mu = self.net_mu.call(x)
        # mu = np.asarray(mu)[0]

        # # Predict sigma
        # x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        # sg = self.net_sg.call(x)
        # sg = np.asarray(sg)[0]

        # # Predict correlations
        # x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        # cr = self.net_cr.call(x)
        # cr = np.asarray(cr)[0]

        # # Define pdf
        # if (self.pdf == 'es'):
        #     pdf = tfd.Normal(mu, sg)
        # if (self.pdf == 'cma-diag'):
        #     pdf = tfd.MultivariateNormalDiag(mu, sg)
        # if (self.pdf == 'cma-full'):
        #     pdf = self.get_cov_pdf(mu, sg, cr)

        # Draw actions
        ac = pdf.sample(n)
        ac = np.asarray(ac)
        ac = np.clip(ac, -1.0, 1.0)

        return ac

    # Train networks
    def train_networks(self):

        # self.train_loop_sg()
        # self.train_loop_cr()
        # self.train_loop_mu()

    # Train loop
    def train_loop(self):

        # # Loop on epochs
        # for epoch in range(self.mu_epochs):

        #     obs, act, adv, n_gen = self.get_history(self.mu_gen)
        #     done = False
        #     btc  = 0

        #     # Visit all available history
        #     while not done:

        #         start    = btc*self.mu_batch*self.n_ind
        #         end      = min((btc+1)*self.mu_batch*self.n_ind,len(obs))
        #         btc     += 1
        #         if (end == len(obs)): done = True
        #         btc_obs  = obs[start:end]
        #         btc_act  = act[start:end]
        #         btc_adv  = adv[start:end]

        #         self.train_mu(btc_obs, btc_adv, btc_act)

    # Train network
    @tf.function
    def train(self, obs, adv, act):

        # var = self.net_mu.trainable_variables
        # with tf.GradientTape() as tape:
        #     tape.watch(var)

        #     cr = tf.convert_to_tensor(self.net_cr.call(obs), tf.float64)
        #     sg = tf.convert_to_tensor(self.net_sg.call(obs), tf.float64)
        #     mu = tf.convert_to_tensor(self.net_mu.call(obs), tf.float64)

        #     loss = self.get_loss(obs, adv, act, mu, sg, cr)

        # grads = tape.gradient(loss, var)
        # norm  = tf.linalg.global_norm(grads)
        # self.net_mu.opt.apply_gradients(zip(grads, var))

    # Compute loss
    def get_loss(self, obs, adv, act, mu, sg, cr):

        # # Compute pdf
        # if (self.pdf == 'es'):
        #     pdf = tfd.Normal(mu[0], sg[0])
        #     log = tf.reduce_sum(pdf.log_prob(act), axis=1)
        # if (self.pdf == 'cma-diag'):
        #     pdf = tfd.MultivariateNormalDiag(mu[0], sg[0])
        #     log = pdf.log_prob(act)
        # if (self.pdf == 'cma-full'):
        #     pdf = self.get_cov_pdf(mu[0], sg[0], cr[0])
        #     log = pdf.log_prob(act)

        # Compute loss
        s     = tf.multiply(adv, log)
        loss  =-tf.reduce_mean(s)

        return loss
