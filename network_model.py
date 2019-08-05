import numpy  as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, parameters):
        self.save_path = parameters.save_path

        self.radius = parameters.radius
        self.primary_number = parameters.primary_number
        self.secondary_number = parameters.secondary_number
        self.CR_router_number = parameters.CR_router_number

        self.user_power_max=parameters.user_power_max
        self.user_power_min=parameters.user_power_min
        self.power_set_number = parameters.power_set_number
        self.noise_power=parameters.noise_power
        self.mu= parameters.mu
        self.sigma= parameters.sigma
        self.channel_gain= parameters.channel_gain
        self.primary_rate_min= parameters.primary_rate_min
        self.secodary_rate_min= parameters.secodary_rate_min
        self.primary_init_power=parameters.primary_init_power

        self.network=np.load(self.save_path + 'network_model.npy',allow_pickle=True)
        self.power_set=np.linspace(self.user_power_min,
                              self.user_power_max,self.power_set_number)

    def create_network(self):
        '''产生模型'''
        '''Primary'''
        length=np.random.uniform(self.radius*(3/4), self.radius,self.primary_number)
        angel=np.random.uniform(0, 2*np.pi, self.primary_number)
        x1=length*np.cos(angel)
        y1= length * np.sin(angel)

        length=np.random.uniform(self.radius*(3/4), self.radius,self.primary_number)
        angel=np.random.uniform(0, 2*np.pi, self.primary_number)
        x2=length*np.cos(angel)
        y2= length * np.sin(angel)

        primary_coord=[x1,y2,x2,y2]

        '''Secodary'''
        length = np.random.uniform(self.radius * (1 / 4), self.radius*(2/4), self.secondary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.secondary_number)
        x1 = length * np.cos(angel)
        y1 = length * np.sin(angel)
        length = np.random.uniform(self.radius * (1 / 4), self.radius*(2/4), self.secondary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.secondary_number)
        x2 = length * np.cos(angel)
        y2 = length * np.sin(angel)
        secondary_coord = [x1, y2, x2, y2]

        '''CR-router'''
        length = np.random.uniform(self.radius * (1 / 5), self.radius, self.CR_router_number)
        angel = np.random.uniform(0, 2 * np.pi, self.CR_router_number)
        x1 = length * np.cos(angel)
        y1 = length * np.sin(angel)
        length = np.random.uniform(self.radius * (1 / 5), self.radius, self.CR_router_number)
        angel = np.random.uniform(0, 2 * np.pi, self.CR_router_number)
        x2 = length * np.cos(angel)
        y2 = length * np.sin(angel)
        CR_router_coord = [x1, y2, x2, y2]


        '''保存数据'''
        network_model = [primary_coord,secondary_coord,CR_router_coord]
        np.save(self.save_path + 'network_model',network_model)

        self.network =network_model

        return primary_coord,secondary_coord,CR_router_coord

    def plot_network(self):
        ''''绘图'''
        primary_coord=self.network[0]
        secondary_coord=self.network[1]
        CR_router_coord=self.network[2]

        plt.figure()
        plt.plot(primary_coord[0],primary_coord[1],
                 'r<',MarkerSize=10,label=u'PU_Tx')
        plt.plot(primary_coord[2],primary_coord[3],
                 'rs',MarkerSize=10,label=u'PU_Rx')

        plt.plot(secondary_coord[0], secondary_coord[1],
                 'g>', MarkerSize=10,label=u'SU_Tx')
        plt.plot(secondary_coord[2], secondary_coord[3],
                 'gs', MarkerSize=10,label=u'SU_Rx')

        plt.plot(CR_router_coord[0], CR_router_coord[1],
                 'bo', MarkerSize=10,label=u'CR-router')

        plt.legend(fontsize=12)

        plt.xlabel(u'x',fontsize=15)
        plt.ylabel(u'y',fontsize=15)
        plt.show()


    def channelgain(self,coord1,coord2,pu_i,su_j):

        '''取出坐标'''
        Tx1 = coord1[0][pu_i]
        Ty1 = coord1[1][pu_i]
        Rx1 = coord1[2][pu_i]
        Ry1 = coord1[3][pu_i]

        Tx2 = coord2[0][su_j]
        Ty2 = coord2[1][su_j]
        Rx2 = coord2[2][su_j]
        Ry2 = coord2[3][su_j]

        # print('coord1:', coord1)
        # print('coord2:', coord2)
        #
        # print('Tx1:',Tx1)
        # print('Ty1:', Ty1)
        # print('Rx1:',Rx1)
        # print('Ry1:', Ry1)
        #
        # print('Tx2:', Tx2)
        # print('Ty2:', Ty2)
        # print('Rx2:', Rx2)
        # print('Ry2:', Ry2)

        '''传输链路距离'''
        dis_Tx1_Rx1 = np.sqrt( (Tx1 - Rx1) ** 2  + (Ty1 - Ry1) ** 2)

        '''干扰链路距离'''
        dis_Tx2_Rx1 = np.sqrt((Tx2 - Rx1) ** 2 + (Ty2- Ry1) ** 2)

        '''信道增益计算'''
        Gain_Tx1_Rx1 = np.power(dis_Tx1_Rx1, self.channel_gain)
        Gain_Tx2_Rx1 = np.power(dis_Tx2_Rx1, self.channel_gain)

        # '''传输链路'''
        # dis_SU_Tx_SU_Rx = np.sqrt(
        #     (coord2[0][su_j] - coord2[2][su_j]) ** 2
        #     + (coord2[1][su_j] - coord2[3][su_j]) ** 2)
        #
        # '''干扰链路'''
        # dis_SU_Rx_PU_Tx = np.sqrt(
        #     (coord2[2][su_j] -coord1[0][pu_i]) ** 2
        #     + (coord2[3][su_j] -coord1[1][pu_i]) ** 2)



        # gain_SU_Tx_SU_Rx = np.power(dis_SU_Tx_SU_Rx, self.channel_gain)
        # gain_SU_Rx_PU_Tx = np.power(dis_SU_Rx_PU_Tx, self.channel_gain)

        return [Gain_Tx1_Rx1,Gain_Tx2_Rx1]

    def envirement(self,action_choose):
        # Gain=self.channelgain(pu_i,su_j)
        done=False
        reward=0

        pu_power = self.primary_init_power

        '''根据动作选择 SU 功率'''
        su_power = self.power_set[action_choose]

        '''根据动作得到状态'''
        new_states=self.CR_router_sensed_power(self.primary_init_power,su_power)

        '''SU 信干噪比检测'''
        SINR_pu = pu_power / (su_power + self.noise_power)
        SINR_su = su_power /(pu_power+ self.noise_power)

        # print('action_choose:', action_choose)
        # print('pu_power:', type(self.primary_init_power))
        # print('su_power:', type(su_power))

        if SINR_pu>=self.primary_rate_min and SINR_su>=self.secodary_rate_min:
            done=True  #到达最好的状态
            reward=10

        # print('pu_power:', self.primary_init_power)
        # print('su_power:', su_power)
        # print('SINR_pu:', SINR_pu)
        # print('SINR_su:', SINR_su)

        '''计算下一次的 PU 功率'''
        pu_new_power=(pu_power*self.primary_rate_min)/SINR_pu
        #超过最大功率
        if pu_new_power<self.user_power_max:
            selected=int((pu_new_power/self.user_power_max)*self.power_set_number)
            # print('selected:',selected)
            pu_new_power = self.power_set[selected]

        elif pu_new_power>self.user_power_max:
            pu_new_power=self.user_power_max

        '''PU 功率更新'''
        self.primary_init_power=pu_new_power
        # print('primary_init_power:',self.primary_init_power,'su_power:',su_power)


        return new_states,reward,done#,SINR_pu,SINR_su

    def Model_based_power_control(self,action_choose):

        done=False
        reward=0

        pu_power = self.primary_init_power

        '''根据动作选择 SU 功率'''
        su_power = self.power_set[action_choose]

        '''根据动作得到状态'''
        new_states=self.CR_router_sensed_power(self.primary_init_power,su_power)

        '''SU 信干噪比检测'''
        SINR_pu = pu_power / (su_power + self.noise_power)
        SINR_su = su_power /(pu_power+ self.noise_power)

        # print('action_choose:', action_choose)
        # print('pu_power:', type(self.primary_init_power))
        # print('su_power:', type(su_power))

        if SINR_pu>=self.primary_rate_min and SINR_su>=self.secodary_rate_min:
            done=True  #到达最好的状态
            reward=10

        # print('pu_power:', self.primary_init_power)
        # print('su_power:', su_power)
        # print('SINR_pu:', SINR_pu)
        # print('SINR_su:', SINR_su)

        '''计算下一次的 PU 功率'''
        pu_new_power=(pu_power*self.primary_rate_min)/SINR_pu
        #超过最大功率
        if pu_new_power<self.user_power_max:
            selected=int((pu_new_power/self.user_power_max)*self.power_set_number)
            # print('selected:',selected)
            pu_new_power = self.power_set[selected]

        elif pu_new_power>self.user_power_max:
            pu_new_power=self.user_power_max

        '''PU 功率更新'''
        self.primary_init_power=pu_new_power
        # print('primary_init_power:',self.primary_init_power,'su_power:',su_power)



        return new_states,reward,done,pu_power,su_power


    def CR_router_sensed_power(self,pu_power,su_power):

        primary_coord = self.network[0]
        secondary_coord = self.network[1]
        CR_router_coord = self.network[2]

        #噪声
        # np.random.seed(2)  #2,8

        # router_noise_power=np.random.rand(self.CR_router_number) * self.sigma
        # print('router_noise_power:', (router_noise_power))

        power_state=[]
        power_pu=[]
        power_su=[]
        for i in range(self.CR_router_number):
            channel_gain1=self.channelgain(CR_router_coord , primary_coord, i, 0)
            channel_gain2 =self.channelgain(CR_router_coord, secondary_coord, i, 0)

            ''''CR-router接收到的用户功率'''
            power_pu.append(pu_power*channel_gain1[1])
            power_su.append(su_power*channel_gain2[1])

            ''''汇总功率状态值'''
            sigma=(pu_power*channel_gain1[1]+su_power*channel_gain2[1])/10
            power_state.append(pu_power*channel_gain1[1] +su_power*channel_gain2[1]+sigma)


        # print('power_pu:',power_pu)
        # print('power_su:',power_su)
        # print('router_noise_power:', router_noise_power)
        #
        # print('power_received:', power_received)

        power_state=np.array(power_state)
        power_state=power_state.reshape(-1)

        return  power_state

    def reset_action(self):
        ''''随机选择动作'''

        random_action_su=np.random.randint(0,len(self.power_set),1)
        power_su=self.power_set[random_action_su]


        random_action_pu=np.random.randint(0,len(self.power_set),1)
        power_pu=self.power_set[random_action_pu]
        self.primary_init_power=power_pu

        # print('random_action_su:',random_action_su)
        # print('power_su:', power_su)
        # print('random_action_pu:',random_action_pu)
        # print('power_pu:', power_pu)

        return power_pu,power_su


