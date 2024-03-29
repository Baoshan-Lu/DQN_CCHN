import os
import torch
import numpy as np
from  model  import DQN
import argparse
import time
from train_model import  Model_train
import network_model

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''DQN_model'''
    parameters = argparse.ArgumentParser()
    parameters.add_argument('--epsion', type=int, default=0.7, help="epsion")
    parameters.add_argument('--targetnet_update_rate', type=int, default=100, help="targetnet_update_rate")
    parameters.add_argument('--memory_capacity', type=int, default=400, help="memory_capacity")
    parameters.add_argument('--batchsize', type=int, default=256)
    parameters.add_argument('--gamma', type=float, default=0.9)
    parameters.add_argument('--epoch', type=int, default=10000)
    parameters.add_argument('--save_path', type=str, default='results/')
    parameters.add_argument('--test_number', type=int, default=100)
    parameters.add_argument('--start_train', type=int, default=300)
    '''CCHN_network'''
    parameters.add_argument('--radius', type=int, default=200)
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
    parameters.add_argument('--pu_power_mode', type=int, default=2)
    parameters.add_argument('--transition_steps', type=int, default=20)


    parameters.add_argument('--gpu_type', type=bool, default=False, choices=[True, False])
    parameters.add_argument('--pretrain', type=bool, default=False, choices=[True, False])

    '''初始参数'''
    parameters = parameters.parse_args(
        ['--CR_router_number','10', '--power_set_number',
         '8','--reward', '10','--batchsize','256',
         '--memory_capacity','400','--start_train', '300',
         '--sigma_factor','3','--test_number', '50',
         '--primary_rate_min', '1.2', '--secodary_rate_min', '0.7',
         '--noise_power', '0.01', '--transition_steps', '20',
          '--epoch','300','--learning_rate','0.001'])


    '''随机种子'''
    np.random.seed(20)
    torch.manual_seed(1)

    '''CCHN网络'''
    cchn=network_model.Network(parameters)
    cchn.create_network()
    cchn.plot_network()
    # for t in range(8):
    new_states, reward, done=cchn.envirement(1)



    '''模型训练'''
    Model_train=Model_train(parameters)
    Model_train.model_trainning()

    # Model_train.secondary_power()
    Model_train.accuracy(200)
