# bc.py: implementation of behavioral cloning in tensorflow
# Author: Nishanth Koganti
# Date: 2018/12/17

# import modules
import os
import gym
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description='behavioral cloning')
    parser.add_argument('expert_data', type=str, help='path to dataset')
    parser.add_argument('env', type=str, help='openai gym env to evaluate')
    parser.add_argument('-r','--render', action='store_true', default=False,
                        help='flag to enable env rendering')
    parser.add_argument('-n','--num_rollouts', type=int, default=5, 
                        help='number of rollouts for testing')
    parser.add_argument('-b','--batch_size', type=int, default=64, 
                        help='batch size for bc training')
    args = parser.parse_args()

    # load expert data
    with open(args.expert_data, 'rb') as f:
        expert_data = pickle.load(f, pickle.HIGHEST_PROTOCOL)

    # parse dataset
    act_data = expert_data['actions']
    obs_data = expert_data['observations']

    # get dimensionality
    act_dim = act_data.shape[1]
    obs_dim = obs_data.shape[1]
    n_samples = obs_data.shape[0]

    # initialize environment environment
    env = gym.make(args.env)
    max_steps = env.spec.timestep_limit

    # loop for num_rollouts
    returns = []
    for i in range(args.num_rollouts):
        print('Iter', i)

        steps = 0
        totalr = 0.0
        done = False
        obs = env.reset()

        while not done:
            action = policy(obs[None, :])
            obs, r, done, _ = env.step(action)

            steps += 1
            totalr += r

            if args.render():
                env.render()

            if steps >= max_steps:
                break
        returns.append(totalr)

    # output statistics
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()