# import modules
import os
import numpy as np
import matplotlib.pyplot as plt

# import utilities
import utils
from timer import timeit
from logger import logger
from model_based_policy import ModelBasedPolicy

class ModelBasedRL(object):

    def __init__(self, env, num_init_random_rollouts=10, max_rollout_length=500,
                 num_onplicy_iters=10, num_onpolicy_rollouts=10,
                 training_epochs=60, training_batch_size=512,
                 render=False, mpc_horizon=15, num_random_action_selection=4096,
                 nn_layers=1):

        # initialize class parameters
        self._env = env
        self._render = render
        self._training_epochs = training_epochs
        self._num_onpolicy_iters = num_onplicy_iters
        self._max_rollout_length = max_rollout_length
        self._training_batch_size = training_batch_size
        self._num_onpolicy_rollouts = num_onpolicy_rollouts

        # create initial random dataset
        logger.info('Gathering random dataset')
        self._random_dataset = self._gather_rollouts(utils.RandomPolicy(env),
                                                     num_init_random_rollouts)

        # create policy instance
        logger.info('Creating policy')
        self._policy = ModelBasedPolicy(env, self._random_dataset, 
                        horizon=mpc_horizon,
                        num_random_action_selection=num_random_action_selection)

        # start timing
        timeit.reset()
        timeit.start('total')

    def _gather_rollouts(self, policy, num_rollouts):

        # initialize dataset class
        dataset = utils.Dataset()

        # loop for num_rollouts
        for _ in range(num_rollouts):

            # reset gym env
            t = 0
            done = False
            state = self._env.reset()

            # generate gym rollout
            while not done:
                # perform rendering
                if self._render:
                    timeit.start('render')
                    self._env.render()
                    timeit.stop('render')

                # get action using MPC
                timeit.start('get action')
                action = policy.get_action(state)
                timeit.stop('get action')

                # step through environment
                timeit.start('env step')
                next_state, reward, done, _ = self._env.step(action)
                timeit.stop('env step')

                # add experience to dataset
                done = done or (t >= self._max_rollout_length)
                dataset.add(state, action, next_state, reward, done)

                # update state variable
                t += 1
                state = next_state

        return dataset

    def _train_policy(self, dataset):

        # timing for policy training
        timeit.start('train policy')

        losses = []

        # loop for self._training_epochs
        for _ in range(self._training_epochs):

            # iterate over dataset
            for states, actions, next_states, _, _ in \
                    dataset.random_iterator(self._training_batch_size):

                # compute loss
                loss = self._policy.train_step(states, actions, next_states)
                losses.append(loss)

        # perform logging
        logger.record_tabular('TrainingLossStart', losses[0])
        logger.record_tabular('TrainingLossFinal', losses[-1])
        timeit.stop('train policy')

    def _log(self, dataset):
        # stop timing
        timeit.stop('total')

        # print logging information
        dataset.log()
        logger.dump_tabular(print_func=logger.info)
        logger.debug('')
        for line in str(timeit).split('\n'):
            logger.debug(line)

        # reset timing
        timeit.reset()
        timeit.start('total')

    def run_q1(self):

        logger.info('Training policy....')
        self._train_policy(self._random_dataset)

        logger.info('Evaluating predictions...')
        for r_num, (states, actions, _, _, _) in enumerate(self._random_dataset.rollout_iterator()):

            pred_states = []
            for state, action in zip(states, actions):
                pred_states.append(self._policy.predict(state, action))

            states = np.asarray(states)
            pred_states = np.asarray(pred_states)

            state_dim = states.shape[1]
            rows = int(np.sqrt(state_dim))
            cols = state_dim // rows

            f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            f.suptitle('Model predictions (red) versus ground truth (black)')

            for i, (ax, state_i, pred_state_i) in enumerate(zip(axes.ravel(), \
                    states.T, pred_states.T)):

                ax.set_title('state {0}'.format(i))
                ax.plot(state_i, color='k')
                ax.plot(pred_state_i, color='r')
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            f.savefig(os.path.join(logger.dir, 
                      'prediction_{0:03d}.jpg'.format(r_num)), 
                      bbox_inches='tight')

        logger.info('All plots saved to folder')

    def run_q2(self):
        logger.info('Random policy')
        self._log(self._random_dataset)

        logger.info('Training policy....')
        self._train_policy(self._random_dataset)

        logger.info('Evaluating policy...')
        eval_dataset = self._gather_rollouts(self._policy, 
                                             self._num_onpolicy_rollouts)

        logger.info('Trained policy')
        self._log(eval_dataset)

    def run_q3(self):
        dataset = self._random_dataset

        itr = -1
        logger.info('Iteration {0}'.format(itr))
        logger.record_tabular('Itr', itr)
        self._log(dataset)

        for itr in range(self._num_onpolicy_iters + 1):
            logger.info('Iteration {0}'.format(itr))
            logger.record_tabular('Itr', itr)

            logger.info('Training policy...')
            self._train_policy(dataset)

            logger.info('Gathering rollouts...')
            new_dataset = self._gather_rollouts(self._policy, 
                                                self._num_onpolicy_rollouts)

            logger.info('Appending dataset...')
            dataset.append(new_dataset)

            self._log(new_dataset)
