import numpy as np
import random

r = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])

q = np.zeros([6, 6], np.float)

greed = 0.8

episode = 0

while episode < 1000:
    state = np.random.randint(0, 5)
    if state != 5:
        next_state_list = []
        for i in range(6):
            if r[state, i] != -1:
                next_state_list.append(i)
        if len(next_state_list) > 0:
            next_state = next_state_list[random.randint(0, len(next_state_list) - 1)]
            q[state, next_state] = r[state, next_state] + greed * max(q[next_state])
    episode = episode + 1
    if episode % 100 == 0:
        print(episode)

i = 0
while i < 5:
    state = i
    i = i + 1
    print("robot 处于{}位置".format(state))
    count = 0
    list = []
    while state != 5:
        if count > 11:
            print("failed ! \n")
            break
        list.append(state)
        next_state = q[state].argmax()
        count = count + 1
        state = next_state
    list.append(5)
    print('path is :')
    print(list)
