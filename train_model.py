import os
import torch
import numpy as np
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
       self.test_number = parameters.test_number
       self.start_train=parameters.start_train

       self.primary_rate_min = parameters.primary_rate_min
       self.secodary_rate_min = parameters.secodary_rate_min

       self.save_path=parameters.save_path
       self.learning_rate = parameters.learning_rate
       self.gpu_type = parameters.gpu_type
       self.memory_capacity = parameters.memory_capacity


       self.epoch=parameters.epoch
       self.learning_rate = parameters.learning_rate
       self.CR_router_number = parameters.CR_router_number
       self.power_set_number=parameters.power_set_number

       print('==========System parameters==========')
       print('CR-routers=', self.CR_router_number, '|  Power_actions=', self.power_set_number)
       print('Pretrained model=', self.pretrain, '| CUDA_use =', self.gpu_type)
       print('Memory_capacity=', self.memory_capacity, '| Minimum batch size=', self.batchsize)
       print('Total epoches=', self.epoch, '|  Learning_rate=', self.learning_rate)
       print('Pu_SINR_min=', self.primary_rate_min, '|  Su_SINR_min=', self.secodary_rate_min)


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

        print('\nStarting training...')

        t0 = time.time()
        metrics = {'time': [], 'epoch': [], 'loss': [], 'reward': []}

        count=0
        loss = 100

        '''初始化，PU,SU随机选择一个动作，计算CR_router感知的功率，作为初始状态值'''
        PU_power, SU_power = self.cchn.reset_action()
        # print('Count:', count, 'PU_power:', PU_power, 'SU_power:', SU_power)
        for epoch in range(self.epoch):

            # '''初始化，PU,SU随机选择一个动作，计算CR_router感知的功率，作为初始状态值'''
            # PU_power,SU_power=self.cchn.reset_action()

            s=self.cchn.CR_router_sensed_power(PU_power,SU_power,self.sigma_factor)
            reward = 0
            # while True:

            '''将初始状态输入eval_net，结合利用e_greedy得到下一个动作'''
            a = dqn.choose_action(s,epoch/self.epoch)

            '''采取此动作,Pu调整自己的功率，计算回报，得到下一个状态'''
            s_, r, done=self.cchn.envirement(a)

            '''收集经验'''
            dqn.store_transition(s, a, r, s_)
            reward += r

            '''经验收集完毕，从经验库中抽取minbatch 来训练'''
            if dqn.memory_counter > self.start_train:  #收集经验之后，开始学习
                loss=dqn.learn()
                # if done: #表示达到最佳状态
                #     if epoch%100==0:
                #         print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Epoch: ', epoch,
                #               '| Reward: %.2f'%reward,'| Loss: %.10f'%loss)

            '''完成任务，那就跳出，直接进入下一个状态'''
            if done:
                '''重新初始化'''
                PU_power, SU_power = self.cchn.reset_action()
                # print('Count:', count, 'PU_power:', PU_power, 'SU_power:', SU_power)
            else:
                s=s_

            count = count + 1


            '''训练过程数据记录'''
            tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            metrics['time'].append(tim1)
            metrics['epoch'].append(int(epoch))
            metrics['loss'].append(float(loss))
            metrics['reward'].append(float(reward))
            json.dump({'metrics': metrics}, fp=open(self.save_path + 'training_process' + '.rs', 'w'), indent=4)

            '''模型保存以及测试'''
            if epoch%100 == 0:
                torch.save(dqn, self.save_path +'dqn_model')
                torch.save(dqn.eval_net, self.save_path + 'eval_net')
                torch.save(dqn.target_net, self.save_path + 'target_net')
                # print('Model saved...')

                #测试一下模型
                Success_rate,SINR_pu,SINR_su,max_transion_step=self.accuracy(self.test_number)

                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      '| Epoch:',epoch,'| Loss: %.10f'%loss,
                      '| Successful_access_rate: ',Success_rate,
                      '| Transion_step: ', max_transion_step,

                      # '\nSINR_pu: ',SINR_pu,'| SINR_su: ',SINR_su
                      )

        t1 = time.time() - t0
        print('Total training time: ', t1)


    def secondary_power(self,search_epoch):

        model=torch.load(self.save_path + 'eval_net')

        # print('\nStarting test...')
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
        # print('\nPU_power_init:', PU_power, ',| SU_power_init', SU_power)

        optimal=0
        max_transion_step=search_epoch
        pu_power1, su_power1=PU_power, SU_power

        for epoch in range(search_epoch):

            '''找到Q值最大的动作'''
            actions_value = model.forward(s)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()

            '''计算reward'''
            s_, r, done,pu_power,su_power,SINR_pu,SINR_su = self.cchn.Model_based_power_control(action)
            # print('Epoch:',epoch,' action:',action,' reward:',r)#'s:',s,,'\ns_:',s_
            # print('\nInput_state:', s)
            # print('Output_actions:', actions_value)
            # print('Choose action: ', action)
            # print('Done: ', done)
            # print('Next_state: ',s_)


            '''达到最佳的状态，跳出'''
            if done:
                # print('PU_power_optimal:', pu_power, 'SU_power_optimal:', su_power)
                optimal=1
                pu_power1, su_power1 =pu_power,su_power
                max_transion_step=epoch+1
                break

            s=s_
            s = torch.unsqueeze(torch.FloatTensor(s), 0)
            if self.gpu_type == True:
                s = s.cuda()

        # print('SINR_pu:', SINR_pu, ',| SU_power_init', SINR_su)

        return  optimal,pu_power1,su_power1,SINR_pu,SINR_su,max_transion_step


    def accuracy(self,Number_of_tests):
        count=0

        '''最大的状态转移次数'''
        transition_steps=20

        Average_transion=[]

        for search_epoch in range(Number_of_tests):
            # print('\nTest:',search_epoch)
            optimal,pu_power,su_power,SINR_pu,SINR_su,max_transion_step=\
                self.secondary_power(transition_steps)

            Average_transion.append(max_transion_step)

            if optimal==1:
                # print('Succeed access...')
                # print('Optimal solution: ',',| PU_power_optimal:', pu_power, ',| SU_power_optimal:', su_power)
                count+=1
            # else:
            #     print( 'Fail access...')
        # print('Successful access rate= %0.2f'% (count/times))

        return  (count/Number_of_tests),SINR_pu,SINR_su,np.mean(Average_transion)
