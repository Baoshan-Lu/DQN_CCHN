import os
import torch
import numpy as np
import gym
from  model  import DQN
import argparse
import time

from train_model import  Model_train
import network_model

env = gym.make('CartPole-v0')
env = env.unwrapped


if __name__ == '__main__':
    np.random.seed(20)  #2,8
    torch.manual_seed(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                                  int) else env.action_space.sample().shape  # to confirm the shape
    '''DQN_model'''
    parameters = argparse.ArgumentParser()
    parameters.add_argument('--epsion', type=int, default=0.8, help="epsion")
    parameters.add_argument('--ENV_A_SHAPE', type=int, default=ENV_A_SHAPE, help="ENV_A_SHAPE")
    parameters.add_argument('--targetnet_update_rate', type=int, default=100, help="targetnet_update_rate")
    parameters.add_argument('--memory_capacity', type=int, default=400, help="memory_capacity")
    parameters.add_argument('--batchsize', type=int, default=256)
    parameters.add_argument('--gamma', type=float, default=0.9)
    parameters.add_argument('--epoch', type=int, default=100)
    parameters.add_argument('--save_path', type=str, default='results/')

    '''CCHN_network'''
    parameters.add_argument('--radius', type=int, default=500)
    parameters.add_argument('--primary_number', type=int, default=1)
    parameters.add_argument('--secondary_number', type=int, default=1)
    parameters.add_argument('--CR_router_number', type=int, default=3)
    parameters.add_argument('--power_set_number', type=int, default=10)
    parameters.add_argument('--user_power_max', type=float, default=0.4)
    parameters.add_argument('--user_power_min', type=float, default=0.05)
    parameters.add_argument('--reward', type=float, default=100)
    parameters.add_argument('--sigma_factor', type=float, default=10)
    parameters.add_argument('--noise_power', type=float, default=0.01)
    parameters.add_argument('--mu', type=float, default=0.0)
    parameters.add_argument('--sigma', type=float, default=0.0005)
    parameters.add_argument('--channel_gain', type=int, default=-1)
    parameters.add_argument('--primary_rate_min', type=float, default=1.2)
    parameters.add_argument('--secodary_rate_min', type=float, default=0.7)
    parameters.add_argument('--primary_init_power', type=float, default=0.1)
    parameters.add_argument('--learning_rate', type=float, default=1 * 1e-3)

    parameters.add_argument('--gpu_type', type=bool, default=False, choices=[True, False])
    parameters.add_argument('--pretrain', type=bool, default=True, choices=[True, False])

    parameters = parameters.parse_args(
        ['--CR_router_number','3', '--power_set_number','10',
         '--learning_rate','0.001','--sigma_factor','3',
         '--memory_capacity','2000','--batchsize','256',
         '--reward', '100', '--epoch','2000',#'--gpu_type', '1', '--pretrain','0'
         ])

    cchn=network_model.Network(parameters)
    cchn.create_network()
    cchn.plot_network()

    Model_train=Model_train(parameters)
    Model_train.model_trainning()

    # Model_train.secondary_power()
    # Model_train.accuracy(100)
