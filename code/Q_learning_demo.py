import numpy as np
# 将Q矩阵初始化为0
q = np.matrix(np.zeros([6, 6]))
# 报酬矩阵为提前定义好的
#-1表示无相连接的边，100为最后通向出口，0表示有连接。
r = np.matrix([[-1, -1, -1, -1,  0,  -1], 
               [-1, -1, -1,  0, -1, 100], 
               [-1, -1, -1,  0, -1,  -1], 
               [-1,  0,  0, -1,  0,  -1], 
               [ 0, -1, -1,  0, -1, 100], 
               [-1,  0, -1, -1,  0, 100]])
#折扣因子（γ）
gamma = 0.8
#是否选择最后策略的概率
e= 0.4
# the main training loop
for time in range(500):
    # random initial state
    state = np.random.randint(0, 6)
    # 如果不是最终转态
    while (state != 5): 
        # 选择可能的动作
        possible_actions = []
        possible_q = []
        for action in range(6):
            if r[state, action] >= 0:
                possible_actions.append(action)
                possible_q.append(q[state, action])
        action = -1
        if np.random.random() < e:
            # 随意选择
            action = possible_actions[np.random.randint(0, len(possible_actions))]
        else:
            action = possible_actions[np.argmax(possible_q)]
        # 更新
        q[state, action] = r[state, action] + gamma * q[action].max()
        #下一步
        state = action
    # 输出训练过程
    if time % 10 == 0:
        print("------------------------------------------------")
        print("训练的次数为: %d" % time)
        print(q)
