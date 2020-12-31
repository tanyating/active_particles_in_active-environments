# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class simple:
    
    def __init__(self, Fx, Fy, v0=10, D=0, Dr=0.1, dt=0.01, T=100000, N=2000, L=20, H=5, h=1, v_wall=0, seed=0):
        np.random.seed(seed)
        
        # v0: constant velocity
        # D: diffusion coefficient
        # Dr: rotational diffusion coefficient
        # dt: time interval
        # T: number of time steps
        # N: number of particles
        # Fx: force along x-axis given potential
        # Fy: force along y-axis given potential
        
        self.v0 = v0
        self.D = D
        self.Dr = Dr
        self.dt = dt
        self.T = T
        self.N = N
        self.L = L
        self.Fx = Fx
        self.Fy = Fy
        self.H = H
        self.h = h
        self.v_wall = v_wall
        
        self.x = np.random.uniform(0, 0, (self.N, ))
        self.y = np.random.uniform(0, 0, (self.N, ))
        self.theta = np.zeros(self.N)
        
        self.X = np.zeros((self.N, int(self.T/100)))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, int(self.T/100)))
        self.Y[:, 0] = self.y
        self.J = np.zeros(self.T)
        
        #self.dX = np.zeros((self.N, self.T))
        #self.dY = np.zeros((self.N, self.T))
        
        for i in range(1, self.T):
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            tmp1 = self.Fx(self.x-self.v_wall*i*self.dt, self.y, self.H, self.h, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += tmp1
            #self.x %= self.L
            tmp2 = self.Fy(self.x-self.v_wall*i*self.dt, self.y, self.H, self.h, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += tmp2
            self.J[i] = np.mean(tmp1) / self.dt
            #self.X[:, i] = self.x
            #self.Y[:, i] = self.y
            if (i%100 == 0):
                self.X[:, int(i/100)] = self.x
                self.Y[:, int(i/100)] = self.y
            
    def local_flux_avg(self, x0, epsilon=0.1):
        last = self.X.shape[1]
        dX_cur = self.dX[:, last-50000:last][np.all([self.X[:, last-50000:last] >= x0-epsilon, self.X[:, last-50000:last] <= x0+epsilon], axis=0)] / (self.dt)
        if (self.Fx == Fx2):
            height = 10 - 2*np.cos(np.pi * x0 / 10)
        else:
            height = 10
        #print("height at %d: %s" % (x0, height))
        return np.mean(dX_cur) * height / (2 * epsilon)
                
    


 
class sym_amplitude:
    
    def __init__(self, v0=1, Dr=1, dt=0.01, T=100000, a=1, D=0, L=20, N=1000):
        
        self.dt = dt
        self.T = T
        self.a = a
        self.D = D
        self.L = L
        self.N = N
        self.v0 = v0
        self.Dr = Dr
    
        self.x = np.random.uniform(0, 20, (self.N,))
        self.y = np.random.uniform(-5, 5, (self.N,))
        
        self.theta = np.zeros(self.N)
        
        self.X = np.zeros((self.N, self.T))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, self.T))
        self.Y[:, 0] = self.y
        
        

        # u(x, y) = (y - a*cos(pi x / 10))^4 / 4
        def Fx(x, y, a):
            return - a * np.pi * (y - a * np.cos(np.pi * x / 10))**3 * np.sin(np.pi * x / 10) / 10
        def Fy(x, y, a):
            return - (y - a * np.cos(np.pi * x / 10))**3

        self.MSDx = np.zeros(self.T)
        self.MSDy = np.zeros(self.T)
        self.MSD = np.zeros(self.T)

        for i in range(1, self.T):
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            self.x += Fx(self.x, self.y, self.a) * self.dt + self.v0 * np.cos(self.theta) * self.dt
            self.y += Fy(self.x, self.y, self.a) * self.dt + self.v0 * np.sin(self.theta) * self.dt
            
            tmp1 = np.square(self.x)
            tmp2 = np.square(self.y)
            
            self.X[:, i] = self.x
            self.Y[:, i] = self.y
            
            self.MSDx[i] = np.mean(tmp1)
            self.MSDy[i] = np.mean(tmp2)
            self.MSD[i] = np.mean(tmp1+tmp2)


           
class asymmetric:
    
    def __init__(self, s1, s2, v0=1, D=0, Dr=0.1, dt=0.01, T=200000, N=1000, L=20, c=10):
        
        # v0: constant velocity
        # s1: slope for first part
        # s2: (-s2) slope for second part
        # D: diffusion coefficient
        # Dr: rotational diffusion coefficient
        # dt: time interval
        # T: number of time steps
        # N: number of particles
        # Fx: force along x-axis given potential
        # Fy: force along y-axis given potential
        # c: stiffness of the walls (coefficient in front of potential)
        
        self.v0 = v0
        self.D = D
        self.Dr = Dr
        self.dt = dt
        self.T = T
        self.N = N
        self.L = L
        self.s1 = s1
        self.s2 = s2
        self.c = c
        
        self.x = np.random.uniform(0, 20, (self.N,))
        self.y = np.random.uniform(-5, 5, (self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        self.X = np.zeros((self.N, self.T))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, self.T))
        self.Y[:, 0] = self.y
        
        self.dX = np.zeros((self.N, self.T))
        self.dY = np.zeros((self.N, self.T))
        
        # total flux every 10000 iterations
        self.J = np.zeros(((int)(self.T / 10000), ))
        
        def Fx(x, y, s1, s2, c):
           if (s1==0):
               a1 = x < 10
               a2 = x >= 10
           else:
               #a1 = x < (20*s2 / (s1+s2))
               #a2 = x >= (20*s2 / (s1+s2))
               a1 = np.all([x < (20*s2 / (s1+s2)) - 0.5, x >= 0.5], axis=0)
               a2 = np.all([x >= (20*s2 / (s1+s2)) + 0.5, x < 19.5], axis=0)
               a3 = np.all([x < (20*s2 / (s1+s2)) - 0.5, x >= 0.5], axis=0)
               
           tmp1 = np.zeros(x[a1].shape[0])
           tmp2 = np.zeros(x[a2].shape[0])
           tmp = np.zeros(x.shape[0])

           b1_1 = y[a1] > s1*x[a1]+5
           b1_2 = y[a2] > -s2*x[a2]+20*s2+5
           b2_1 = y[a1] < -s1*x[a1]-5
           b2_2 = y[a2] < s2*x[a2]-20*s2-5

           tmp1[b1_1] = -2 * (s1**2) * x[a1][b1_1] + 2 * s1 * y[a1][b1_1] - 2 * s1 * 5
           tmp2[b1_2]= -2 * (s2**2) * x[a2][b1_2] - 2 * s2 * y[a2][b1_2] + 40 * (s2**2) + 2 * s2 * 5
           tmp1[b2_1]= -2 * (s1**2) * x[a1][b2_1] - 2 * s1 * y[a1][b2_1] - 2 * s1 * 5
           tmp2[b2_2] = -2 * (s2**2) * x[a2][b2_2] + 2 * s2 * y[a2][b2_2] + 40 * (s2**2) + 2 * s2 * 5

           tmp[a1] = tmp1
           tmp[a2] = tmp2

           return tmp*c
            
        def Fy(x, y, s1, s2, c):
            if (s1==0):
                a1 = x < 10
                a2 = x >= 10
            else:
                a1 = x < (20*s2 / (s1+s2))
                a2 = x >= (20*s2 / (s1+s2))
            tmp1 = np.zeros(x[a1].shape[0])
            tmp2 = np.zeros(x[a2].shape[0])
            tmp = np.zeros(x.shape[0])

            b1_1 = y[a1] > s1*x[a1]+5
            b1_2 = y[a2] > -s2*x[a2]+20*s2+5
            b2_1 = y[a1] < -s1*x[a1]-5
            b2_2 = y[a2] < s2*x[a2]-20*s2-5

            tmp1[b1_1] = 2 * s1 * x[a1][b1_1] - 2 * y[a1][b1_1] + 2*5
            tmp2[b1_2] = -2 * s2 * x[a2][b1_2] - 2 * y[a2][b1_2] + 40 * s2 + 2*5
            tmp1[b2_1] = -2 * s1 * x[a1][b2_1] - 2 * y[a1][b2_1] - 2 * 5
            tmp2[b2_2] = 2 * s2 * x[a2][b2_2] - 2 * y[a2][b2_2] - 40 * s2 - 2*5

            tmp[a1] = tmp1
            tmp[a2] = tmp2

            return tmp*c
        
        
        for i in range(1, self.T):
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            self.dX[:, i] = Fx(self.x, self.y, self.s1, self.s2, self.c) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += self.dX[:, i]
            self.x %= self.L
            self.dY[:, i] = Fy(self.x, self.y, self.s1, self.s2, self.c) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += self.dY[:, i]
            self.X[:, i] = self.x
            self.Y[:, i] = self.y
            if ((i+1)%10000 == 0):
                j = (int)((i+1-10000) / 10000)
                self.J[j] = np.mean(self.dX[:, i+1-10000:i+1]) / self.dt
                print("instant total flux: %f" % self.J[j])
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            
    def plot_traj(self):
        f = plt.figure(figsize=(5,3))
        c1 = 'b'
        c2 = 'r'
        if (self.v0 == 0):
            c1 = 'r'
            c2 = 'b'
        plt.plot(self.X[0], self.Y[0], '%so'%(c1), alpha=0.2)
        plt.title("a single trajectory")
        if (self.s1 + self.s2 == 0):
            x1 = np.linspace(0, 10)
            x2 = np.linspace(10, 20)
        else:
            x1 = np.linspace(0, 20*self.s2/(self.s1+self.s2))
            x2 = np.linspace(20*self.s2/(self.s1+self.s2), 20)
        plt.plot(x1, self.s1*x1+5, c2)
        plt.plot(x2, -self.s2*x2+20*self.s2+5, c2)
        plt.plot(x1, -self.s1*x1-5, c2)
        plt.plot(x2, self.s2*x2-20*self.s2-5, c2)
        #plt.savefig('asym_traj_v0=%d.png'%(self.v0))
        
    def local_flux_avg(self, x0, epsilon=0.1):
        last = self.X.shape[1]
        dX_cur = self.dX[:, last-50000:last][np.all([self.X[:, last-50000:last] >= x0-epsilon, self.X[:, last-50000:last] <= x0+epsilon], axis=0)] / (self.dt)
        if (self.s1 == 0 and self.s2 == 0):
            height = 10
        elif (x0 < (20*self.s2 / (self.s1+self.s2))):
            height = 2 * self.s1 * x0 + 10
        else:
            height = -2 * self.s2 * x0 + 40 * self.s2 + 10
        print("height at %d: %s" % (x0, height))
        return np.mean(dX_cur) * height / (2 * epsilon)
 
    
class ratchet:
    
    def __init__(self, v0=10, D=0, Dr=0.1, dt=0.01, T=200000, N=2000, L=20, c=2, p=4, A=0.3):
        
        # v0: constant velocity
        # s1: slope for first part
        # s2: (-s2) slope for second part
        # D: diffusion coefficient
        # Dr: rotational diffusion coefficient
        # dt: time interval
        # T: number of time steps
        # N: number of particles
        # Fx: force along x-axis given potential
        # Fy: force along y-axis given potential
        # c: stiffness of the walls (coefficient in front of potential)
        
        self.v0 = v0
        self.D = D
        self.Dr = Dr
        self.dt = dt
        self.T = T
        self.N = N
        self.L = L
        self.p = p
        self.A = A
        self.c = c
        
        self.x = np.random.uniform(0, self.L, (self.N,))
        self.y = np.random.uniform(-5, 5, (self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        self.X = np.zeros((self.N, self.T))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, self.T))
        self.Y[:, 0] = self.y
        
        self.dX = np.zeros((self.N, self.T))
        self.dY = np.zeros((self.N, self.T))
        
        # total flux every 10000 iterations
        self.J = np.zeros(((int)(self.T / 10000), ))
        
        # potential: V = c((y-0)/5)^p(1+A Exp[Cos[2 Pi x/L] ]Sin[2 Pi x/L])
        
        def Fx(x, y, c, p, A, L):
           return -(5**(-p))*c*(y**p)*(2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.cos(2* np.pi * x / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * (np.sin(2 * np.pi * x /L)**2)) / L
            
        def Fy(x, y, c, p, A, L):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * x / L)) * np.sin(2 * np.pi * x / L))
        
        
        for i in range(1, self.T):
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            self.dX[:, i] = Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += self.dX[:, i]
            self.x %= self.L
            self.dY[:, i] = Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += self.dY[:, i]
            self.X[:, i] = self.x
            self.Y[:, i] = self.y
            if ((i+1)%10000 == 0):
                j = (int)((i+1-10000) / 10000)
                self.J[j] = np.mean(self.dX[:, i+1-10000:i+1]) / self.dt
                print("instant total flux: %f" % self.J[j])
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            


def plot_traj(x, y):
    f = plt.figure(figsize=(5,3))
    plt.plot(x, y, 'ro', alpha=0.2)
    plt.title("a single trajectory")

def rel_msd(dX, X0, v, t):
    X = dX.cumsum(axis=1)
    return np.square((X.T-X0).T - v*t).mean(axis=0)

def MSD(dX):
    return np.square(dX.cumsum(axis=1)).mean(axis=0)

def plot_msd(msd, dt):
    f = plt.figure(figsize=(10,3))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    t = np.arange(msd.shape[0]) * dt
    
    ax.plot(t, msd)
    ax.set_xlabel("t")
    ax.set_ylabel("MSD")
    
    ax2.loglog(t[1:], msd[1:])
    ax2.set_title("log-log plot for MSD versus t") 
    
    
    

# flux
def local_flux_sum(x0, dX, X, dt, epsilon):
    last = X.shape[1]
    dX_cur = dX[:, last-50000:last][np.all([X[:, last-50000:last] >= x0-epsilon, X[:, last-50000:last] <= x0+epsilon], axis=0)] / (50000 * dt)
    return np.sum(dX_cur) / (2 * epsilon)

def total_flux_at_t(t, dX, dt):
    return np.mean(np.sum(dX[:, t] / dt))

def total_flux(dX, dt):
    flux = np.zeros(10)
    for i in range(10):
        dX_cur = dX[:, i*10000:(i+1)*10000]
        flux[i] = np.mean(dX_cur) / dt
    avg_flux = np.mean(flux)
    err = np.std(flux)
    f = plt.figure(figsize=(5,3))
    plt.plot(flux, 'r+')
    plt.title("total flux at increasing time intervals")
    return avg_flux, err


# several Fx, Fy corresponding with symmetric potential

def Fx1(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > 5+np.cos(np.pi * x / 10)
    b2 = y < -5+np.cos(np.pi * x / 10)
    tmp[b1] = (-np.pi*(y[b1]-np.cos(np.pi*x[b1]/10)-5)*np.sin(np.pi*x[b1]/10)/5)
    tmp[b2] = (-np.pi*(y[b2]-np.cos(np.pi*x[b2]/10)+5)*np.sin(np.pi*x[b2]/10)/5)
    return tmp*100

def Fy1(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > 5+np.cos(np.pi * x / 10)
    b2 = y < -5+np.cos(np.pi * x / 10)
    tmp[b1] = (-2*y[b1]+2*np.cos(np.pi*x[b1]/10)+10)
    tmp[b2] = (-2*y[b2]+2*np.cos(np.pi*x[b2]/10)-10)
    return tmp*100



def Fx5(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H/2+h*np.cos(2 * np.pi * x / L)
    b2 = y < -H/2-h*np.cos(2 * np.pi * x / L)
    tmp[b1] = -4*(y[b1]-(H/2+h*np.cos(2 * np.pi * x[b1] / L)))**3*h*np.sin(2 * np.pi * x[b1] / L)* 2 * np.pi / L
    # -2 (y-f(x)) h sin(2pi x/L)*2 pi/L
    tmp[b2] = -4*((-H/2-h*np.cos(2 * np.pi * x[b2] / L))-y[b2])**3*h*np.sin(2 * np.pi * x[b2] / L)* 2 * np.pi / L
    return tmp*2

def Fy5(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H/2+h*np.cos(2 * np.pi * x / L)
    b2 = y < -H/2-h*np.cos(2 * np.pi * x / L)
    tmp[b1] = -4*(y[b1]-(H/2+h*np.cos(2 * np.pi * x[b1] / L)))**3
    tmp[b2] = 4*((-H/2-h*np.cos(2 * np.pi * x[b2] / L))-y[b2])**3
    return tmp*2

def Fx6(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H/2+h*np.cos(2 * np.pi * x / L)
    #b2 = y < -H/2
    tmp[b1] = -4*(y[b1]-(H/2+h*np.cos(2 * np.pi * x[b1] / L)))**3*h*np.sin(2 * np.pi * x[b1] / L)* 2 * np.pi / L
    # -2 (y-f(x)) h sin(2pi x/L)*2 pi/L
    #tmp[b2] = -2*((-H/2)-y[b2])
    return tmp*2

def Fy6(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H/2+h*np.cos(2 * np.pi * x / L)
    b2 = y < -H/2
    tmp[b1] = -4*(y[b1]-(H/2+h*np.cos(2 * np.pi * x[b1] / L)))**3
    tmp[b2] = 4*((-H/2)-y[b2])**3
    return tmp*2

def Fx7(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H+h*np.cos(2 * np.pi * x / L)
    #b2 = y < -H/2
    tmp[b1] = -4*(y[b1]-(H+h*np.cos(2 * np.pi * x[b1] / L)))**3*h*np.sin(2 * np.pi * x[b1] / L)* 2 * np.pi / L
    # -2 (y-f(x)) h sin(2pi x/L)*2 pi/L
    #tmp[b2] = -2*((-H/2)-y[b2])
    return tmp*2

def Fy7(x, y, H, h, L):
    tmp = np.zeros(x.shape[0])
    b1 = y > H+h*np.cos(2 * np.pi * x / L)
    b2 = y < 0
    tmp[b1] = -4*(y[b1]-(H+h*np.cos(2 * np.pi * x[b1] / L)))**3
    tmp[b2] = 4*(0-y[b2])**3
    return tmp*2



def Fx2(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > 5-np.cos(np.pi * x / 10)
    b2 = y < -5+np.cos(np.pi * x / 10)
    tmp[b1] = (np.pi*(y[b1]+np.cos(np.pi*x[b1]/10)-5)*np.sin(np.pi*x[b1]/10)/5)
    tmp[b2] = (-np.pi*(y[b2]-np.cos(np.pi*x[b2]/10)+5)*np.sin(np.pi*x[b2]/10)/5)
    return tmp*100

def Fy2(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > 5-np.cos(np.pi * x / 10)
    b2 = y < -5+np.cos(np.pi * x / 10)
    tmp[b1] = (-2*y[b1]-2*np.cos(np.pi*x[b1]/10)+10)
    tmp[b2] = (-2*y[b2]+2*np.cos(np.pi*x[b2]/10)-10)
    return tmp*100


def Fx3(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > x+5
    b2 = y < -x-5
    
    tmp[b1] = -2 * x[b1] + 2 * y[b1] - 2 * 5
    tmp[b2]= -2 * x[b2] - 2 * y[b2] - 2 * 5
 
    return tmp*100
        
def Fy3(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > x+5
    b2 = y < -x-5

    tmp[b1] = 2 * x[b1] - 2 * y[b1] + 2*5
    tmp[b2] = -2 * x[b2] - 2 * y[b2] - 2 * 5

    return tmp*100


def Fx4(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > -x+25
    b2 = y < x-25
    
    tmp[b1]= -2 * x[b1] - 2 * y[b1] + 40 + 2 * 5
    tmp[b2] = -2 * x[b2] + 2 * y[b2] + 40 + 2 * 5
    
    return tmp*10

def Fy4(x, y):
    tmp = np.zeros(x.shape[0])
    b1 = y > -x+25
    b2 = y < x-25
    
    tmp[b1] = -2 * x[b1] - 2 * y[b1] + 40 + 2*5
    tmp[b2] = 2 * x[b2] - 2 * y[b2] - 40 - 2*5
    
    return tmp*10