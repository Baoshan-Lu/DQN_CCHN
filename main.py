import os
import torch
import numpy as np
import gym
from  model  import DQN
import argparse
import time

import tools
import network_model

env = gym.make('CartPole-v0')
env = env.unwrapped


if __name__ == '__main__':
    np.random.seed(8)  #2,8
    torch.manual_seed(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                                  int) else env.action_space.sample().shape  # to confirm the shape
    '''DQN_model'''
    parameters = argparse.ArgumentParser()
    parameters.add_argument('--states', type=int, default=4)
    parameters.add_argument('--actions', type=int, default=2)
    parameters.add_argument('--epsion', type=int, default=0.8, help="epsion")
    parameters.add_argument('--ENV_A_SHAPE', type=int, default=ENV_A_SHAPE, help="ENV_A_SHAPE")
    parameters.add_argument('--targetnet_update_rate', type=int, default=100, help="targetnet_update_rate")
    parameters.add_argument('--memory_capacity', type=int, default=1000, help="memory_capacity")
    parameters.add_argument('--batchsize', type=int, default=32)
    parameters.add_argument('--learning_rate', type=float, default=0.01)
    parameters.add_argument('--gamma', type=float, default=0.9)
    parameters.add_argument('--epoch', type=int, default=100)
    parameters.add_argument('--gpu_type', type=bool, default=False, choices=[True, False])
    parameters.add_argument('--save_path', type=str, default='results/')

    '''CCHN_network'''
    parameters.add_argument('--primary_number', type=int, default=1)
    parameters.add_argument('--secondary_number', type=int, default=1)
    parameters.add_argument('--CR_router_number', type=int, default=80)
    parameters.add_argument('--radius', type=int, default=500)

    parameters.add_argument('--user_power_max', type=int, default=0.4)
    parameters.add_argument('--user_power_min', type=int, default=0.05)
    parameters.add_argument('--power_set_number', type=int, default=8)




    args = parameters.parse_args(['--epoch','100'])

    # tools.train(args)
    cchn=network_model.Network(args)

    cchn.create_network()
    cchn.plot_network()
