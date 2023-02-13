"""
This file handles parsing and validation of the cli arguments to the train_rainbow.py file.
If left unspecified, some argument defaults are set dynamically here.
"""

import argparse
import distutils
import random
import socket
from copy import deepcopy


def read_args():
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser(description=('Training framework for Rainbow DQN\n'
                                                 '  - individial components of Rainbow can be adjusted with cli args (below)\n'),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cnn_body_filters",type=int,default=12)
    parser.add_argument("--cnn_head_filters",type=int,default=2)
    parser.add_argument("--cnn_hex_size",type=int,default=5)
    parser.add_argument("--cnn_zero_fill",type=parse_bool,default=False)
    parser.add_argument("--cnn_mode",type=parse_bool,default=False)
    parser.add_argument('--roundrobin_players', type=int, default=10, help='How many player will play in each roundrobin elo tournament')
    parser.add_argument('--roundrobin_games', type=int, default=12, help='How many games will be played in each matchup of the roundrobin tournament')

    # training settings
    parser.add_argument('--training_frames', type=int, default=100_000_000, help='train for n environment interactions ("game_frames" in the code)')
    parser.add_argument('--record_every', type=int, default=60*50, help='wait at least x seconds between episode recordings (default is to use environment specific presets)')
    parser.add_argument('--seed', type=int, default=0, help='seed for pytorch, numpy, environments, random')
    parser.add_argument('--use_wandb', type=parse_bool, default=True, help='whether use "weights & biases" for tracking metrics, video recordings and model checkpoints')
    parser.add_argument('--use_amp', type=parse_bool, default=False, help='whether to enable automatic mixed precision for the forward passes')
    parser.add_argument('--decorr', type=parse_bool, default=True, help='try to decorrelate state/progress in parallel envs')
    parser.add_argument('--num_required_repeated_actions', type=int, default=20)
    parser.add_argument('--hex_size',type=int,default=5)
    parser.add_argument('--norm', type=parse_bool, default=False, help='Use norm in graph net')
    parser.add_argument('--load_model',type=str,default=None)
    parser.add_argument('--model_name',type=str,default="two_headed")
    parser.add_argument('--num_layers',type=int,default=13)
    parser.add_argument('--num_head_layers',type=int,default=2)
    parser.add_argument('--hidden_channels',type=int,default=32)
    parser.add_argument('--prune_exploratories',type=parse_bool, default=True)
    parser.add_argument('--grow',type=parse_bool, default=False)
    parser.add_argument('--testing_mode',type=parse_bool, default=False)


    parser.add_argument('--wandb_tag', type=str, default=None, help='')

    # dqn settings
    parser.add_argument('--buffer_size', type=int, default=int(2 ** 17), help='capacity of experience replay buffer (must be a power of two)')
    parser.add_argument('--burnin', type=int, default=100_000, help='how many transitions should be in the buffer before start of training')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor')
    parser.add_argument('--sync_dqn_target_every', type=int, default=32_000, help='sync Q target net every n frames')

    parser.add_argument('--batch_size', type=int, default=256, help='sample size when sampling from the replay buffer')
    parser.add_argument('--parallel_envs', type=int, default=64, help='number of envs in the vectorized env')
    parser.add_argument('--train_count', type=int, default=2, help='how often to train on a batch_size batch for every step (of the vectorized env)')
    parser.add_argument('--subproc_vecenv', type=parse_bool, default=True, help='whether to run each environment in it\'s own subprocess (always enabled for gym-retro)')

    parser.add_argument('--double_dqn', type=parse_bool, default=True, help='whether to use the double-dqn TD-target')
    parser.add_argument('--prioritized_er', type=parse_bool, default=True, help='whether to use prioritized experience replay')
    parser.add_argument('--prioritized_er_beta0', type=float, default=0.45, help='importance sampling exponent for PER (0.4 for rainbow, 0.5 for dopamine)')
    parser.add_argument('--prioritized_er_time', type=int, default=None, help='time period over which to increase the IS exponent (+inf for dopamine; default is value of training_frames)')
    parser.add_argument('--n_step', type=int, default=1, help='the n in n-step bootstrapping')
    parser.add_argument('--init_eps', type=float, default=1.0, help='initial dqn exploration epsilon (when not using noisy-nets)')
    parser.add_argument('--final_eps', type=float, default=0.01, help='final dqn exploration epsilon (when not using noisy-nets)')
    parser.add_argument('--eps_decay_frames', type=int, default=500_000, help='exploration epsilon decay frames, 250_000 for rainbow paper, 1M for dopamine (when not using noisy-nets)')
    parser.add_argument('--noisy_dqn', type=parse_bool, default=False, help='whether to use noisy nets dqn')
    parser.add_argument('--noisy_sigma0', type=float, default=0.5, help='sigma_0 parameter for noisy nets dqn')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for adam (0.0000625 for rainbow paper/dopamine, 0.00025 for DQN/procgen paper)')
    parser.add_argument('--lr_decay_steps', type=int, default=None, help='learning rate is decayed every n game_steps (disabled by default)')
    parser.add_argument('--lr_decay_factor', type=float, default=None, help='factor by which lr is multiplied (disabled by default)')
    parser.add_argument('--adam_eps', type=float, default=0.00015, help='epsilon for adam (0.00015 for rainbow paper/dopamine, 0.0003125 for DQN/procgen paper); default is to use 0.005/batch_size')
    parser.add_argument('--max_grad_norm', type=float, default=10, help='gradient will be clipped to ensure its l2-norm is less than this')
    parser.add_argument('--loss_fn', type=str, default='huber', help='loss function ("mse" or "huber")')

    args = parser.parse_args()

    # some initial checks to ensure all arguments are valid
    assert (args.sync_dqn_target_every % args.parallel_envs) == 0 # otherwise target may not be synced since the main loop iterates in steps of parallel_envs
    assert args.loss_fn in ('mse', 'huber')
    assert (args.lr_decay_steps is None) == (args.lr_decay_factor is None)
    assert args.burnin > args.batch_size

    args.user_seed = args.seed

    # apply default values if user did not specify custom settings
    if args.adam_eps is None: args.adam_eps = 0.005/args.batch_size
    if args.prioritized_er_time is None: args.prioritized_er_time = args.training_frames

    # turn off e-greedy exploration if noisy_dqn is enabled
    if args.noisy_dqn:
        args.init_eps = 0.002
        args.final_eps = 0.0
        args.eps_decay_frames = 20000

    # clean up the parameters that get logged to wandb
    args.instance = socket.gethostname()
    wandb_log_config = deepcopy(vars(args))
    del wandb_log_config['record_every']
    del wandb_log_config['use_wandb']
    del wandb_log_config['wandb_tag']

    return args, wandb_log_config
