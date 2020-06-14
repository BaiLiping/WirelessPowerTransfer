'''
author: Bai Liping
date: Jun 15, 2020
'''

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

def build_code_book(M,N,f_c,start_position):
    code_book_startposition=[]
    theta_base=math.pi/2*N #the range is only pi/2 divided by N
    for i in range(N):
        f=compute_bf_vector(M,start_position+theta_base*i,f_c)
        code_book_startposition.append(f)
    return code_book_startposition


class WPTEnv(discrete.DiscreteEnv):
    '''
        Environment:
        L transmitting nodes, each with M radiating element, beaming energy to
        K receivers. N beamforming vectors in codebook. with different f_c comes
        different pass loss.

        Observation:
            Type: Box(6 or 8)
            Num Observation                                    Min      Max
            0   received energy on receiver 1                   0j       3j
            1   received energy on receiver 2                   0j       3j
            .          .                                        .        .
            k-1 received energy on receiver k                   0j       3j
            k   choise made by transmitter1                     0        3
            k+1   choice made by transmitter2                   0        3
            k+2   choice made by transmitter3                   0        3

        Actions:
           Range of Adjustment: pi/4 to 3pi/4
           Type: Discrete
           Num  Action
           0    pi/4
           1    pi/4+pi/2self.N
           2    pi/4+2pi/2self.N
           .     .
           N-1  pi/4+(N-1)pi/2self.N
           N    3pi/4

    '''
    metadata={'render.modes':['human']}
    def __init__(self):
        print('A Radio Environment Has been Initialized')
        self.M= 16 # number of radiating element
        self.K=2 #number of Receivers
        self.L=4 #number of transmitting nodes
        self.N=4 #discretization of pi each slice is 45 degree
        self.f_c=4e7 #frequency
        self.length_of_signal=10000
        self.x=
        self.node_position=[(0,0),(0,30),(30,30),(30,0)]
        code_book=[]
        for i in range(self.L):
            start_position=math.pi/4+math.pi*i
            code_book.append(build_code_book(self.M,self.N,self.f_c,start_position))
        self.code_book=code_book
        self.count=0
        self.coices=np.zeros((self.L,self.M))#register the choices
        self.signal=np.zeros((self.L,self.K,self.length_of_signal))
        self.energy=np.zeros(self.K)
        self.receiver_position=generate_receivers(self.K)
        print('There are {} Energe Receivers'.format(self.K))
        print(self.receiver_position)
        print(self.code_book[1])
        self.num_actions=self.N
        bound_lower=[]
        bound_higher=[]
        for i in range(self.K):
            bound_lower.append(0)
            bound_higher.append(3)
        for j in range(self.L):
            bound_lower.append(0)
            bound_higher.append(3)
        bounds_lower = np.array(bound_lower)
        bounds_upper = np.array(bound_higher)

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here

        self.seed(seed=1)
        state=[]
        for i in range(self.K):
            state.append(0)
        for j in range(self.L):
            state.append(0)
        self.state = np.array(state)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        # Initialize f_n of both cells
        self.receiver_position=generate_receivers(self.K)
        self.state=None
        self.step_count = 0

        return np.array(self.state)

     def step(self, action):
         done=None
         info=None
         choose_L=self.count % self.N
         F=self.code_book[choose_L]
         f=F[action]
         self.choices[choose_L]=f
         total_energy_old=np,sum(self.state[0:self.K])
         total_energy_new=0
         reward=0
         for j in range(self.K):
             self.signal[choose_L][j]=self.compute_signal(self.receiver_position[j][0],self.receiver_position[j][1],self.node_position[choose_L][0],self.node_position[choose_L][1],f,self.x_t)
             e_j=self._compute_energy(j)
             self.state[j]=e_j
             total_energy_new+=e_j
             if e_j<e_min:
                 print('receiver {} total enegey smaller than min value'.format(j))
                 reward-=50

        if total_energy_new>total_energy_old:
            reward+=100
        elif total_energy_new=total_energy_old:
            break
        else:
            reward-=300
         return np.array(self.state), reward, done, info

    def _path_loss(self, x_rx, y_rx, x_tx, y_tx=0):
        f_c = self.f_c
        c = 3e8 # speed of light
        d = math.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2)
        h_B = 20
        h_R = 1.5
        print('Distance from cell site is: {} m'.format(d))
        # FSPL
        L_fspl = -10*np.log10((4.*math.pi*c/f_c / d) ** 2)
        # COST231
        C = 3
        a = (1.1 * np.log10(f_c/1e6) - 0.7)*h_R - (1.56*np.log10(f_c/1e6) - 0.8)
        L_cost231  = 46.3 + 33.9 * np.log10(f_c/1e6) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d/1000.) + C
        L = L_cost231
        return L # in dB
    def _compute_channel(self, x_rx, y_rx, x_tx, y_tx):
        # theta is the steering angle.  Sampled iid from unif(0,pi).
        theta = ####np.random.uniform(low=0, high=math.pi)

        path_loss_LOS = 10 ** (self._path_loss(x_rx, y_rx, x_tx, y_tx) / 10.)
        alpha= 1. / math.sqrt(path_loss_LOS)
        # initialize the channel as a complex variable.
        h = np.zeros(self.M, dtype=complex)
        a_theta = self._compute_bf_vector(theta)
        h += alpha  * a_theta.T # scalar multiplication into a vector
#        print ('Warning: channel gain is {} dB.'.format(10*np.log10(LA.norm(h, ord=2))))
        return h

    def _compute_signal(self, x_rx, y_rx,x_tx,y_tx,f,x_t):
        h=self._compute_channel(x_rx,y_rx,x_tx,y_tx)
        y_t=np.dot(h.conj(),f)*x_t

        return y_t

    def _compute_energy(self,receiver_number):
        for i in range(self.L):
            y_j=self.signal[i][receiver_number]
            e_j=np.dot(y_i,y_i.conj())

        return e_j
