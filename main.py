import os
import torch
import numpy as np
import gym
from  model  import DQN
import argparse
import time

env = gym.make('CartPole-v0')
env = env.unwrapped


def run(parameters):



    dqn = DQN(parameters)
    if parameters.gpu_type==True:
        dqn=dqn.cuda()

    print('\nCollecting experience...')
    for i_episode in range(parameters.epoch):

        '''初始化一个状态'''
        s = env.reset()
        # time.sleep(5)
        # env.close()
        # print(len(s))
        ep_r = 0
        while True:
            # env.render()
            '''输入状态，利用搜索取得一个期望动作'''
            a = dqn.choose_action(s)

            # take action 采用一个动作
            '''采取此动作，进入下一个状态'''
            s_, r, done, info = env.step(a)

            '''获得回报'''
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            '''收集经验'''
            dqn.store_transition(s, a, r, s_)

            ep_r += r

            '''经验收集完毕，从经验库中抽取minbatch 来训练'''
            if dqn.memory_counter > parameters.memory_capacity:  #收集经验之后，开始学习
                dqn.learn()
                if done: #表示达到最佳状态
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            '''不管经验库收不收集完成，如果已经完成任务，那就跳出，直接进入下一个状态'''
            if done:
                break

            s = s_


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                                  int) else env.action_space.sample().shape  # to confirm the shape

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

    parameters.add_argument('--data_path', type=str, default='../data/')
    parameters.add_argument('--save_path', type=str, default='../results/')

    args = parameters.parse_args(['--epoch','200'])

    run(args)