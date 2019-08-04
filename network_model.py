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

    def create_network(self):
        '''产生模型'''
        '''Primary'''
        length=np.random.uniform(self.radius*(3/4), self.radius,self.primary_number)
        angel=np.random.uniform(0, 2*np.pi, self.primary_number)
        x=length*np.cos(angel)
        y= length * np.sin(angel)
        primary_coord=[x,y]

        '''Secodary'''
        length=np.random.uniform(self.radius*(1/4), self.radius,self.primary_number)
        angel=np.random.uniform(0, 2*np.pi, self.primary_number)
        x=length*np.cos(angel)
        y= length * np.sin(angel)
        secondary_coord=[x,y]

        '''CR-router'''
        length=np.random.uniform(50, self.radius,self.CR_router_number)
        angel=np.random.uniform(0, 2*np.pi, self.CR_router_number)
        CR_router_x=length*np.cos(angel)
        CR_router_y= length * np.sin(angel)
        CR_router_coord=[CR_router_x,CR_router_y]

        # print(primary_coord)
        # print(secondary_coord)
        # print(CR_router_coord)

        '''保存数据'''

        # network_model = {'primary_coord':primary_coord,
        #                  'secondary_coord': secondary_coord,
        #                  'CR_router_coord': CR_router_coord}

        network_model = [primary_coord,secondary_coord,CR_router_coord]

        np.save(self.save_path + 'network_model',network_model)
        # print(primary_coord)
        # print(secondary_coord)
        # print(CR_router_coord)
        return primary_coord,secondary_coord,CR_router_coord

    def plot_network(self):

        ''''绘图'''
        network=np.load(self.save_path + 'network_model.npy',allow_pickle=True)
        primary_coord=network[0]
        secondary_coord=network[1]
        CR_router_coord=network[2]
        # print('primary_coord:',primary_coord)
        # print('secondary_coord:',secondary_coord)
        # print('CR_router_coord:',CR_router_coord)
        # print(network)

        plt.figure()
        plt.plot(primary_coord[0],primary_coord[1],
                 'r<',MarkerSize=10,label=u'PU')
        plt.plot(secondary_coord[0], secondary_coord[1],
                 'gs', MarkerSize=10,label=u'SU')
        plt.plot(CR_router_coord[0], CR_router_coord[1],
                 'bo', MarkerSize=10,label=u'CR-router')
        plt.legend(fontsize=12)

        plt.xlabel(u'x',fontsize=15)
        plt.ylabel(u'y',fontsize=15)
        plt.show()

    def calculate(self):


        pass