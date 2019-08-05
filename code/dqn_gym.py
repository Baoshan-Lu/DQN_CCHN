import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 100
env = gym.make('CartPole-v0')
env = env.unwrapped


N_ACTIONS = env.action_space.n  #2个动作，左右
N_STATES = env.observation_space.shape[0] #状态，4个

# print(N_STATES,N_ACTIONS)

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)   # 4,50
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)  # 50,2
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):

        self.count=0
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0              # 用于 target 更新计时
        self.memory_counter = 0                  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) #优化器选择
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            # print('actions_value=',actions_value)
            # print( torch.max(actions_value, 1))
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))

        # print(self.count, 's=',s,'a=',a, 'r=',r, 's_=',s_,'\ntransition=', transition)
        # print(transition)
        # r # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY

        # print('index=',index)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新 Q现实
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #复制参数
        self.learn_step_counter += 1

        # print('self.learn_step_counter=',self.learn_step_counter)
        # print('self.memory_counter =',self.memory_counter )

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  #随机抽取记忆


        b_memory = self.memory[sample_index, :]

        # print(len(b_memory))
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]) #当前状态,:N_STATES==取前 N_STATES 个状态的大小
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)) #拟采用的动作
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]) #获得的回报
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]) #采取此动作后进去的新的状态，-N_STATES: 倒取最后N_STATES

        # print(b_s.shape,b_a.shape,b_r.shape,b_s_.shape)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)

        '''Q估计:过去的经验值'''
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 输入当期状态，进行估计。shape (batch, 1)

        '''Q现实:真实回报+现实与估计的差距'''
        q_next = self.target_net(b_s_).detach()  # 输入对应新的状态获得真实回报。detach from graph, don't backpropagate 不进行反向传播
        self.count=self.count+1


        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)

        # print(self.count,'\nq_next.max(1)=', q_next,q_next.max(1),'\nq_next.max(1)[0]=',q_next.max(1)[0])
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):

    '''初始化一个状态'''
    s = env.reset()
    # print('state:', s.shape)
    # print(len(s))
    ep_r = 0
    while True:
        # env.render()
        '''输入状态，利用搜索取得一个期望动作'''
        # print('s:',s)
        a = dqn.choose_action(s)



        # take action 采用一个动作
        '''采取此动作，进入下一个状态'''
        s_, r, done, info = env.step(a)
        # print('r:',r)



        '''获得回报'''
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        '''收集经验'''
        dqn.store_transition(s, a, r, s_)
        # print('s:',type(s))
        # print('a:',type(a))
        # print('r:', type(r))
        # print('s_:', type(s_))
        ep_r += r

        '''经验收集完毕，从经验库中抽取minbatch 来训练'''
        if dqn.memory_counter > MEMORY_CAPACITY:  #收集经验之后，开始学习
            dqn.learn()
            if done: #表示达到最佳状态
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        '''不管经验库收不收集完成，如果已经完成任务，那就跳出，直接进入下一个状态'''
        if done:
            break

        s = s_