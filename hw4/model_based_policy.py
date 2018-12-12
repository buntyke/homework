# import modules
import numpy as np
import tensorflow as tf
from IPython.terminal.debugger import set_trace as keyboard

# import utilities
import utils

class ModelBasedPolicy(object):

    def __init__(self, env, init_dataset, horizon=15,
                 num_random_action_selection=4096, nn_layers=1):

        self._horizon = horizon
        self._learning_rate = 1e-3
        self._cost_fn = env.cost_fn
        self._nn_layers = nn_layers
        self._init_dataset = init_dataset
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._state_dim = env.observation_space.shape[0]
        self._num_random_action_selection = num_random_action_selection

        self._sess, self._state_ph, self._action_ph, self._next_state_ph, \
        self._next_state_pred, self._loss, self._optimizer, self._best_action = \
            self._setup_graph()

    def _setup_placeholders(self):

        # initialize place holders
        state_ph = tf.placeholder(tf.float32, shape=(None, self._state_dim))
        action_ph = tf.placeholder(tf.float32, shape=(None, self._action_dim))
        next_state_ph = tf.placeholder(tf.float32, shape=(None, self._state_dim))

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):

        # get dataset statistics
        state_std = self._init_dataset.state_std
        state_mean = self._init_dataset.state_mean

        action_std = self._init_dataset.action_std
        action_mean = self._init_dataset.action_mean

        delta_state_std = self._init_dataset.delta_state_std
        delta_state_mean = self._init_dataset.delta_state_mean

        # normalize input data
        state_norm = utils.normalize(state, state_mean, state_std)
        action_norm = utils.normalize(action, action_mean, action_std)

        # perform delta prediction
        inp = tf.concat([state_norm, action_norm], 1)
        out = utils.build_mlp(inp, self._state_dim, 'policy', reuse=reuse)

        # perdict next state
        next_state_pred = state + utils.unnormalize(out, 
                            delta_state_mean, delta_state_std)
        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):

        # compute state differences
        true_diff = next_state_ph - state_ph
        pred_diff = next_state_pred - state_ph

        # get statistics
        delta_state_std = self._init_dataset.delta_state_std
        delta_state_mean = self._init_dataset.delta_state_mean

        # compute normalized values
        true_diff_norm = utils.normalize(true_diff, delta_state_mean, 
                                         delta_state_std)
        pred_diff_norm = utils.normalize(pred_diff, delta_state_mean, 
                                         delta_state_std)

        # compute loss and initialize optimization
        loss = tf.losses.mean_squared_error(true_diff_norm, 
                                            pred_diff_norm, scope='policy')
        optim = tf.train.AdamOptimizer(self._learning_rate)
        optimizer = optim.minimize(loss)
        return loss, optimizer

    def _setup_action_selection(self, state_ph):

        # initialize random action sequences
        action_sequences = tf.random_uniform((self._horizon, self._num_random_action_selection, 
                                             self._action_dim), 
                                             minval=self._action_space_low, 
                                             maxval=self._action_space_high, 
                                             dtype=tf.float32)

        # initialize variables per sequence
        seq_cost = tf.tile([0.0],[self._num_random_action_selection])
        seq_state = tf.tile(state_ph,[self._num_random_action_selection,1])

        # roll-out sequence
        for t in range(self._horizon):
            seq_next_state = self._dynamics_func(seq_state, action_sequences[t,:,:], reuse=tf.AUTO_REUSE)
            seq_cost += self._cost_fn(seq_state, action_sequences[t,:,:], seq_next_state)
            seq_state = seq_next_state

        min_ind = tf.argmin(seq_cost, 0)
        best_action = action_sequences[0,min_ind,:]
        return best_action

    def _setup_graph(self):

        sess = tf.Session()

        # setup computation graph
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, tf.AUTO_REUSE)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, 
                                               next_state_pred)

        # perform action selection
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):

        # perform loss computation
        optim, loss = self._sess.run([self._optimizer,self._loss], feed_dict={self._state_ph: states, 
                            self._action_ph: actions, self._next_state_ph: next_states})
        
        return loss

    def predict(self, state, action):

        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        # perform next state prediction
        out = self._sess.run(self._next_state_pred, 
                            feed_dict={self._state_ph: state[None,:], self._action_ph: action[None,:]})
        next_state_pred = out[0,:]

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):

        assert np.shape(state) == (self._state_dim,)

        # perform best action selection
        best_action = self._sess.run(self._best_action, 
                             feed_dict={self._state_ph: state[None,:]})

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
