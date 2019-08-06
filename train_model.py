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

class Model_train(object):
    def __init__(self, parameters):

       self.cchn=Network(parameters)
       self.parameters=parameters
       self.pretrain=parameters.pretrain
       self.batchsize = parameters.batchsize
       self.sigma_factor = parameters.sigma_factor


       self.save_path=parameters.save_path
       self.learning_rate = parameters.learning_rate
       self.gpu_type = parameters.gpu_type
       self.memory_capacity = parameters.memory_capacity


       self.epoch=parameters.epoch
       self.learning_rate = parameters.learning_rate
       self.CR_router_number = parameters.CR_router_number
       self.power_set_number=parameters.power_set_number

    def model_trainning(self):
        # cchn = Network(parameters)
        #模型保存路径
        try:
            os.mkdir(self.save_path)
        except Exception:
            pass

        dqn = DQN(self.parameters)

        if self.pretrain==True:
            dqn=torch.load(self.save_path +'dqn_model')


        print('CR-routers=',self.CR_router_number,'|  Power_actions=',self.power_set_number)
        print('Use pretrained model=',self.pretrain,'| Use CUDA=',self.gpu_type)
        print('Memory_capacity=',self.memory_capacity,'| Minimum batch size=',self.batchsize)
        print('Total epoches=',self.epoch,'|  Learning_rate=',self.learning_rate)

        print('\nStarting training...')

        t0 = time.time()
        metrics = {'time': [], 'epoch': [], 'loss': [], 'reward': []}

        count=0
        loss = 100
        for epoch in range(self.epoch):

            '''初始化，PU,SU随机选择一个动作，计算CR_router感知的功率，作为初始状态值'''
            PU_power,SU_power=self.cchn.reset_action()
            # print('PU_power:',PU_power,'SU_power:',SU_power)
            s=self.cchn.CR_router_sensed_power(PU_power,SU_power,self.sigma_factor)
            reward = 0
            while True:
                '''将初始状态输入eval_net，结合利用e_greedy得到下一个动作'''
                a = dqn.choose_action(s,epoch/self.epoch)

                '''采取此动作,Pu调整自己的功率，计算回报，得到下一个状态'''
                s_, r, done=self.cchn.envirement(a)

                '''收集经验'''
                dqn.store_transition(s, a, r, s_)

                reward += r

                # if count%500==0:
                #     print('r=',r)

                '''经验收集完毕，从经验库中抽取minbatch 来训练'''
                if dqn.memory_counter > self.memory_capacity:  #收集经验之后，开始学习

                    loss=dqn.learn()
                    if done: #表示达到最佳状态
                        if epoch%100==0:
                            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Epoch: ', epoch,
                                  '| Reward: %.2f'%reward,'| Loss: %.10f'%loss)

                '''不管经验库收不收集完成，如果已经完成任务，那就跳出，直接进入下一个状态'''
                if done:
                    break

                count = count + 1

                s = s_

                        # 数据保存
            tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            metrics['time'].append(tim1)
            metrics['epoch'].append(int(epoch))
            metrics['loss'].append(float(loss))
            metrics['reward'].append(float(reward))
            json.dump({'metrics': metrics}, fp=open(self.save_path + 'training_process' + '.rs', 'w'), indent=4)


        #模型保存
        if self.pretrain == False:  # 保存预训练模型
            torch.save(dqn, self.save_path +'dqn_model')
            torch.save(dqn.eval_net, self.save_path + 'eval_net')
            torch.save(dqn.target_net, self.save_path + 'target_net')
            print('Model saved...')

        t1 = time.time() - t0
        print('Total training time: ', t1)


    def secondary_power(self,search_epoch):

        model=torch.load(self.save_path + 'eval_net')

        print('\nStarting test...')
        # print('gpu_type=',self.gpu_type,' memory_capacity=',self.memory_capacity,
        #           ' learning_rate=',self.learning_rate,
        #           ' epoch=',self.epoch,' CR-routes=',self.CR_router_number,
        #           ' power_set_number=',self.power_set_number)


        '''随机从一个状态出发'''
        PU_power, SU_power = self.cchn.reset_action()
        s = self.cchn.CR_router_sensed_power(PU_power, SU_power,self.sigma_factor)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if self.gpu_type==True:
            s=s.cuda()
        print('PU_power_init:', PU_power, 'SU_power_init', SU_power)

        optimal=0
        epoch_max=search_epoch
        pu_power1, su_power1=PU_power, SU_power

        for epoch in range(search_epoch):

            '''找到Q值最大的动作'''
            actions_value = model.forward(s)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()

            '''计算reward'''
            s_, r, done,pu_power,su_power = self.cchn.Model_based_power_control(action)
            print('Epoch:',epoch,' action:',action,' reward:',r)#'s:',s,,'\ns_:',s_

            '''达到最佳的状态，跳出'''
            if done:
                # print('PU_power_optimal:', pu_power, 'SU_power_optimal:', su_power)
                optimal=1
                epoch_max=epoch
                pu_power1, su_power1 =pu_power,su_power
                break

        return  optimal,epoch_max,pu_power1,su_power1


    def accuracy(self,times):

        for search_epoch in range(times):
            optimal, epoch_max,pu_power,su_power=self.secondary_power(search_epoch)
            print('Optimal=', optimal, '|  Epoch_max=', epoch_max)
            print('PU_power_optimal:', pu_power, 'SU_power_optimal:', su_power)











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



