import gym
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.toy_text import discrete

def compute_bf_vector(M,theta,f_c):
    c = 3e8 # speed of light
    wavelength = c / f_c
    d = wavelength / 2. # antenna spacing
    k = 2. * math.pi / wavelength
    exponent = 1j * k * d * math.cos(theta) * np.arange(M)
    f = 1. / math.sqrt(M) * np.exp(exponent)
    return f

def generate_receivers(K):
    #generate random field of K receivers
    receiver_positions=np.random.randint(1,29,size=(K,2))
    return receiver_positions

def build_code_book(M,N,f_c):
    code_book=[]
    theta_base=math.pi/N
    for i in range(N):
        f=compute_bf_vector(M,theta_base*i,f_c)
        code_book.append(f)
    return code_book


class WPTEnv(discrete.DiscreteEnv):
    '''
        Observation:
            Type: Box(6 or 8)
            Num Observation                                    Min      Max
            0   received energy on receiver 1                 -1w       1w
            1   received energy on receiver 2                 -1w       1w
            .          .                                        .        .
            k-1 received energy on receiver k                 -1w       1w
            k   choise made by transmitter1                     0        3
            k+1   choice made by transmitter2                   0        3
            k+2   choice made by transmitter3                   0        3

    '''
    metadata={'render.modes':['human']}
    def __init__(self):
        print('A Radio Environment Has been Initialized')
        self.M= 16 # number of radiating element
        self.K=2 #number of Receivers
        self.L=4 #number of transmitting nodes
        self.N=4 #discretization of pi each slice is 45 degree
        self.f_c=4e8 #frequency
        self.receiver_position=generate_receivers(self.K)
        print('There are {} Energe Receivers'.format(self.K))
        print(self.receiver_position)
        self.code_book=build_code_book(self.M, self.N,self.f_c)
        print(self.code_book)
        self.num_actions=8
        bound_lower=[]
        bound_higher=[]
        for i in range(self.K):
            bound_lower.append(-1)
            bound_higher.append(1)
        for j in range(self.L):
            bound_lower.append(0)
            bound_higher.append(3)
        bounds_lower = np.array(bound_lower)
        bounds_upper = np.array(bound_higher)

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here

        self.seed(seed=1)

        self.state = None
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
