import os
import torch
import numpy as np
import gym
from  model  import DQN
import argparse
import time
from datetime import datetime

import json
from json import encoder
import sys

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf8')
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from  network_model  import  Network

def model_trainning(parameters):

    #cchn
    cchn=Network(parameters)

    #模型保存路径
    try:
        os.mkdir(parameters.save_path)
    except Exception:
        pass

    dqn = DQN(parameters)
    if parameters.gpu_type==True:
        dqn=dqn.cuda()

    print('\nCollecting experience...')

    t0 = time.time()
    metrics = {'time': [], 'epoch': [], 'loss': [], 'reward': []}
    
    count=0

    for epoch in range(parameters.epoch):

        '''初始化，PU,SU随机选择一个动作，计算CR_router感知的功率，作为初始状态值'''
        PU_power,SU_power=cchn.reset_action()
        # print('PU_power:',PU_power,'SU_power:',SU_power)
        s=cchn.CR_router_sensed_power(PU_power,SU_power)

        # print('s:',s)


        # s = env.reset()
        Reward = 0
        while True:
            # env.render()

            '''将初始状态输入eval_net，结合利用e_greedy得到下一个动作'''
            a = dqn.choose_action(s)

            # take action 采用一个动作
            '''采取此动作，计算回报，得到下一个状态'''
            # s_, r, done, info = env.step(SU_action)
            s_, r, done=cchn.envirement(a)
            # print('s_: ', s_, '| done: %.2f' % done)

            # '''获得回报'''
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            #
            # print('s:', type(s))
            # print('a:', type(a))
            # print('r:', type(r))
            # print('s_:', type(s_))

            '''收集经验'''
            dqn.store_transition(s, a, r, s_)


            Reward += r

            # if count%100==0:
            #     print('r=',r)

            '''经验收集完毕，从经验库中抽取minbatch 来训练'''
            if dqn.memory_counter > parameters.memory_capacity:  #收集经验之后，开始学习
                loss=dqn.learn()
                if done: #表示达到最佳状态
                    if epoch%100==0:
                        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Epoch: ', epoch,'| Reward: %.2f'%Reward,'| Loss: %.2f'%loss)

            '''不管经验库收不收集完成，如果已经完成任务，那就跳出，直接进入下一个状态'''
            if done:
                break

            count = count + 1

            s = s_

        # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Epoch: ', epoch, '| Reward: %.2f' % Reward,
        #       '| Loss: %.2f' % loss)
        # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Epoch: ', epoch, '| Reward: %.2f' % Reward)
        # '| Loss: %.2f' % loss)

    #模型保存
    torch.save(dqn, parameters.save_path +'dqn_model')
    torch.save(dqn.eval_net, parameters.save_path + 'eval_net')


    # 数据保存
    tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # metrics['time'].append(tim1)
    # metrics['epoch'].append(int(epoch))
    # metrics['loss'].append(float(loss))
    # metrics['reward'].append(float(ep_r))
    # json.dump({'metrics': metrics}, fp=open(parameters.save_path +'training_process'  + '.rs', 'w'), indent=4)

    t1 = time.time() - t0
    print('Total training time: ', t1)

def secondary_power(parameters):

    model=torch.load(parameters.save_path + 'eval_net')
    #初始化一个动作

    #得到一个状态

    #推理出下一个动作

    #循环


    pass

def primary_user_power():
    pass



def choose_action():
    '''选择动作'''
    pass

def obtain_state():
    '''状态更新'''
    pass

def calculate_reward():
    '''回报计算'''
    pass






# env = gym.make('CartPole-v0')
# env = env.unwrapped
#
# def train(parameters):
#
#     #模型保存路径
#     try:
#         os.mkdir(parameters.save_path)
#     except Exception:
#         pass
#
#     dqn = DQN(parameters)
#     if parameters.gpu_type==True:
#         dqn=dqn.cuda()
#
#     print('\nCollecting experience...')
#
#     t0 = time.time()
#     metrics = {'time': [], 'epoch': [], 'loss': [], 'reward': []}
#
#     for epoch in range(parameters.epoch):
#
#         '''初始化一个状态'''
#         s = env.reset()
#         # time.sleep(5)
#         # env.close()
#         # print(len(s))
#         ep_r = 0
#         while True:
#             # env.render()
#             '''输入状态，利用搜索取得一个期望动作'''
#             a = dqn.choose_action(s)
#
#             # take action 采用一个动作
#             '''采取此动作，进入下一个状态'''
#             s_, r, done, info = env.step(a)
#
#             '''获得回报'''
#             x, x_dot, theta, theta_dot = s_
#             r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#             r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#             r = r1 + r2
#
#             '''收集经验'''
#             dqn.store_transition(s, a, r, s_)
#
#             ep_r += r
#
#             '''经验收集完毕，从经验库中抽取minbatch 来训练'''
#             if dqn.memory_counter > parameters.memory_capacity:  #收集经验之后，开始学习
#                 loss=dqn.learn()
#                 if done: #表示达到最佳状态
#                     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Epoch: ', epoch,'| Reward: %.2f'%ep_r,'| Loss: %.2f'%loss)
#
#             '''不管经验库收不收集完成，如果已经完成任务，那就跳出，直接进入下一个状态'''
#             if done:
#                 break
#
#             s = s_
#
#     #模型保存
#     torch.save(dqn, parameters.save_path +'dqn_model')
#     torch.save(dqn.eval_net, parameters.save_path + 'eval_net')
#
#
#     # 数据保存
#     tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#     metrics['time'].append(tim1)
#     metrics['epoch'].append(int(epoch))
#     metrics['loss'].append(float(loss))
#     metrics['reward'].append(float(ep_r))
#     json.dump({'metrics': metrics}, fp=open(parameters.save_path +'training_process'  + '.rs', 'w'), indent=4)
#
#     t1 = time.time() - t0
#     print('Total training time: ', t1)



