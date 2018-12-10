# import general modules
import os
import time
import argparse

# import homework modules
from logger import logger
from model_based_rl import ModelBasedRL
from half_cheetah_env import HalfCheetahEnv

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('question', type=str, choices=('q1, q2, q3'))

parser.add_argument('--render', action='store_true')
parser.add_argument('--nn_layers', type=int, default=1)
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--env', type=str, default='HalfCheetah', choices=('HalfCheetah',))
args = parser.parse_args()

# obtain data directory
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# initialize experiment name
exp_name = '{0}_{1}_{2}'.format(args.env,
                                args.question,
                                args.exp_name if args.exp_name else time.strftime("%d-%m-%Y_%H-%M-%S"))

# create experiment directory
exp_dir = os.path.join(data_dir, exp_name)
assert not os.path.exists(exp_dir),\
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(exp_dir)
os.makedirs(exp_dir, exist_ok=True)

# intialize logger
logger.setup(exp_name, os.path.join(exp_dir, 'log.txt'), 'debug')

# initialize gym env
env = {
    'HalfCheetah': HalfCheetahEnv()
}[args.env]

# initialize model-based rl instance
mbrl = ModelBasedRL(env=env,
                    render=args.render,
                    mpc_horizon=args.mpc_horizon,
                    num_random_action_selection=args.num_random_action_selection,
                    nn_layers=args.nn_layers)

# initialize run function depending on question
run_func = {
    'q1': mbrl.run_q1,
    'q2': mbrl.run_q2,
    'q3': mbrl.run_q3
}[args.question]
run_func()
