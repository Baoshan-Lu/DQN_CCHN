import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self,parameters ):
        super(Net, self).__init__()
        self.state = parameters.CR_router_number
        self.action = parameters.power_set_number

        # self.states=parameters.states
        # self.actions = parameters.actions


        self.hidden1 = nn.Linear(self.state, 256)   # 4,50
        self.hidden1.weight.data.normal_(0, 0.1)   # initialization

        self.hidden2 = nn.Linear(256, 256)   # 4,50
        self.hidden2.weight.data.normal_(0, 0.1)   # initialization

        self.hidden3 = nn.Linear(256, 512)   # 4,50
        self.hidden3.weight.data.normal_(0, 0.1)   # initialization

        self.out = nn.Linear(512, self.action)  # 50,2
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)

        x = self.hidden2(x)
        x = F.relu(x)

        x = self.hidden3(x)
        x = torch.tanh(x)

        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self,parameters):
        self.gpu_type = parameters.gpu_type
        self.save_path = parameters.save_path

        self.epsion = parameters.epsion
        # self.ENV_A_SHAPE = parameters.ENV_A_SHAPE
        self.targetnet_update_rate = parameters.targetnet_update_rate
        self.memory_capacity = parameters.memory_capacity
        self.states = parameters.CR_router_number
        self.action = parameters.power_set_number

        self.pretrain = parameters.pretrain
        self.learning_rate = parameters.learning_rate
        self.batchsize = parameters.batchsize
        self.gamma = parameters.gamma

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((self.memory_capacity, self.states * 2 + 2))  # 初始化记忆库

        self.count = 0

        ''''建立Q网络'''
        self.eval_net, self.target_net = Net(parameters), Net(parameters)
        if self.pretrain==True: #使用预训练模型
            self.eval_net, self.target_net = torch.load(self.save_path + 'eval_net'), \
                                             torch.load(self.save_path + 'eval_net')
        ''''GPU训练'''
        if self.gpu_type == True:
            self.eval_net, self.target_net=self.eval_net.cuda(), self.target_net.cuda()

        ''''优化器选择'''
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate) #优化器选择
        self.loss_func = nn.MSELoss()

    def choose_action(self, x,k):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if self.gpu_type == True:
            x=x.cuda()

        # 这里只输入一个 sample
        if np.random.uniform() < self.epsion*(1-k):   # 选最优动作
            actions_value = self.eval_net.forward(x)

            # print('actions_value=',actions_value)
            # print('torch.max(actions_value, 1)=', torch.max(actions_value, 1))
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            # print('action=', action)
            action = action[0]
            #if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:   # 选随机动作
            action = np.random.randint(0, self.action)
            # action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)

        # print('x============:', x)
        # # print('actions_value:', actions_value)
        # print('chosen_action:', action)

        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))

        # print(self.count, 's=',s,'a=',a, 'r=',r, 's_=',s_,'\ntransition=', transition)
        # print(transition)
        # r # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memory_capacity

        # print('index=',index)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新 Q现实
        if self.learn_step_counter % self.targetnet_update_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # print('self.learn_step_counter=',self.learn_step_counter)
        # print('self.memory_counter =',self.memory_counter )

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memory_capacity, self.batchsize)  #随机抽取记忆


        b_memory = self.memory[sample_index, :]

        # print(len(b_memory))
        b_s = torch.FloatTensor(b_memory[:, :self.states]) #当前状态,:N_STATES==取前 N_STATES 个状态的大小
        b_a = torch.LongTensor(b_memory[:, self.states:self.states+1].astype(int)) #拟采用的动作
        b_r = torch.FloatTensor(b_memory[:, self.states+1:self.states+2]) #获得的回报
        b_s_ = torch.FloatTensor(b_memory[:, -self.states:]) #采取此动作后进去的新的状态，-N_STATES: 倒取最后N_STATES

        if self.gpu_type == True:
            b_s=b_s.cuda()
            b_a=b_a.cuda()
            b_r=b_r.cuda()
            b_s_=b_s_.cuda()
        # print(b_s.shape,b_a.shape,b_r.shape,b_s_.shape)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)

        '''Q估计:过去的经验值'''
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 输入当期状态，进行估计。shape (batch, 1)

        '''Q现实:真实回报+现实与估计的差距'''
        q_next = self.target_net(b_s_).detach()  # 输入对应新的状态获得真实回报。detach from graph, don't backpropagate 不进行反向传播
        self.count=self.count+1

        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batchsize, 1)   # shape (batch, 1)'\nq_next=',q_next,

        # print(self.count,'\nq_eval=', q_eval.view(1, self.batchsize),'\nq_target=',q_target.view(1, self.batchsize))

        # print(self.count,'\nq_next.max(1)=', q_next,q_next.max(1),'\nq_next.max(1)[0]=',q_next.max(1)[0])

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return  loss.cpu().detach().numpy()
