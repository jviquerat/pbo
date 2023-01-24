# Custom imports
from pbo.src.base import *

###############################################
### Class discrete
### A PBO agent for discrete optimization
class discrete(base_pbo):
    def __init__(self, params, act_dim, obs_dim):

        # Initialize from arguments
        self.pdf        = params.pdf
        self.act_dim    = len(act_dim) # number of parameters
        self.obs_dim    = obs_dim
        self.param_dim  = act_dim # list containing the dimension for each param
        self.n_gen      = params.n_gen
        self.n_ind      = params.n_ind
        self.size       = self.n_gen*self.n_ind
        self.n_cpu      = params.n_cpu

        self.param_gen        = params.param_gen
        self.batch            = params.batch
        self.lr               = params.lr
        self.epochs           = params.epochs
        self.adv_clip         = params.adv_clip
        self.adv_decay        = params.adv_decay
        self.continuity_regul = params.continuity_regul

        # Build network
        self.net = nn(params.arch,
                      sum(self.param_dim),
                      'relu',
                      None,
                      self.lr)

        # Init network parameters
        dummy = self.net(tf.ones([1,self.obs_dim]))

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
        self.avg_rwd = np.zeros( self.n_gen,              dtype=np.float64)

    # Get actions from network
    def get_actions(self, state, n):

        # Predict
        x  = tf.convert_to_tensor([state[0]], dtype=tf.float64)
        cat_params = self.net.call(x)
        cat_params = np.asarray(cat_params)[0] #convert to np.array

        # Build pdfs
        pdfs  = [None]*self.act_dim
        index = 0
        for i in range(self.act_dim):
            pdfs[i] = tfd.Categorical(cat_params[index:index + self.param_dim[i]])
            index  += self.param_dim[i]

        # Draw actions
        actions = np.zeros([n,self.act_dim])
        for j in range(n):
            for i,pdf in enumerate(pdfs):
                ac = pdf.sample(1)
                actions[j,i] = ac

        return actions

    # Train networks
    def train_networks(self):

        self.train_loop()

    # Train loop
    def train_loop(self):

        # Loop on epochs
        for epoch in range(self.epochs):

            obs, act, adv, n_gen = self.get_history(self.param_gen)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.batch*self.n_ind
                end      = min((btc+1)*self.batch*self.n_ind,len(obs))
                btc     += 1
                if (end == len(obs)): done = True
                btc_obs  = obs[start:end]
                btc_act  = act[start:end]
                btc_adv  = adv[start:end]

                self.train(btc_obs, btc_adv, btc_act)

    # Train network
    @tf.function
    def train(self, obs, adv, act):

        var = self.net.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(var)

            cat_params = tf.convert_to_tensor(self.net.call(obs), tf.float64)

            loss = self.get_loss(obs, adv, act, cat_params)

        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net.opt.apply_gradients(zip(grads, var))

    # Compute loss
    def get_loss(self, obs, adv, act, cat_params):

        # Build pdf
        cat_params = cat_params[0]
        pdfs       = [None]*self.act_dim
        index      = 0
        regul      = 0
        for i in range(self.act_dim):
            pdfs[i] = tfd.Categorical(cat_params[index:index + self.param_dim[i]])
            probs   = tf.nn.softmax(cat_params[index:index + self.param_dim[i]])

            for j in range(len(probs)-1):
                regul += (probs[j] - probs[j+1])**2

            index += self.param_dim[i]

        # Compute action probability under current policy,
        # decomposing over the action parameters
        # Use independance of the different pdf and log(product) = sum(log)
        log = tf.zeros([act.shape[0]], dtype=tf.dtypes.float64)
        for i in range(self.act_dim):
            log += pdfs[i].log_prob(act[:,i])

        # Compute loss
        s     = tf.multiply(adv, log)
        loss  =-tf.reduce_mean(s)

        regul *= self.continuity_regul
        loss  += regul

        return loss
