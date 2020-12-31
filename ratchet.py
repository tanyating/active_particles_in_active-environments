# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# active particles without any boundary
class nowall:
    
    def __init__(self, v0=1, D=0, Dr=0.1, dt=0.01, T=200000, N=1000, L=20, seed=0):
        np.random.seed(seed)
        
        # v0: constant velocity
        # s1: slope for first part
        # s2: (-s2) slope for second part
        # D: diffusion coefficient
        # Dr: rotational diffusion coefficient
        # dt: time interval
        # T: number of time steps
        # N: number of particles
        
        self.v0 = v0
        self.D = D
        self.Dr = Dr
        self.dt = dt
        self.T = T
        self.N = N
        self.L = L
        
        #self.x = np.random.uniform(0, self.L, (self.N,))
        self.x = np.zeros((self.N,))
        #self.y = np.random.uniform(-5, 5, (self.N,))
        self.y = np.zeros((self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        #self.X = np.zeros((self.N, self.T))
        #self.X[:, 0] = self.x
        #self.Y = np.zeros((self.N, self.T))
        #self.Y[:, 0] = self.y
        
        #self.dX = np.zeros((self.N, self.T))
        #self.dY = np.zeros((self.N, self.T))
        
        # total flux every 10000 iterations
        self.J = np.zeros(self.T)
        self.msd = np.zeros(self.T)
        
        self.X = np.zeros((self.N, int(self.T/1000)))
        self.Y = np.zeros((self.N, int(self.T/1000)))
        
        
        for i in range(1, self.T):
            
            
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            #self.dX[:, i] = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp1 = v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += tmp1
            #self.x %= self.L
            #self.dY[:, i] = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp2 = v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += tmp2
            #self.X[:, i] = self.x
            #self.Y[:, i] = self.y
            self.J[i] = np.mean(tmp1) / self.dt
            self.msd[i] = np.mean((self.x - np.mean(self.x))**2)
            
            if (i%1000 == 0):
                self.X[:, int(i/1000)] = self.x
                self.Y[:, int(i/1000)] = self.y
            
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            
    def anim(self, fold=True):
        fig, ax = plt.subplots(figsize=(5, 3))
        if (fold==True):
            ax.set(xlim=(0, 20), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'bo')[0]
            def animate(i):
                line.set_xdata(self.X[:, i]%self.L)
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('nowall_fold_v0=%d.mp4'%(self.v0))
        else:
            ax.set(xlim=(-2000, 2020), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'ro')[0]
            def animate(i):
                line.set_xdata(self.X[:, i])
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('nowall_v0=%d.mp4'%(self.v0))

class simple:
    
    def __init__(self, v0=1, D=0, Dr=0.1, dt=0.01, T=200000, N=1000, L=20, c=2, p=4, A=0.3, sym=False, seed=0):
        
        np.random.seed(seed)
        
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
        
        #self.x = np.random.uniform(0, self.L, (self.N,))
        self.x = np.zeros((self.N,))
        #self.y = np.random.uniform(-5, 5, (self.N,))
        self.y = np.zeros((self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        #self.X = np.zeros((self.N, self.T))
        #self.X[:, 0] = self.x
        #self.Y = np.zeros((self.N, self.T))
        #self.Y[:, 0] = self.y
        
        #self.dX = np.zeros((self.N, self.T))
        #self.dY = np.zeros((self.N, self.T))
        
        # total flux every 10000 iterations
        self.J = np.zeros(self.T)
        self.msd = np.zeros(self.T)
        # potential: V = c((y-0)/5)^p(1+A Exp[Cos[2 Pi x/L] ]Sin[2 Pi x/L])
        
        def Fx1(x, y, c, p, A, L):
           return -(5**(-p))*c*(y**p)*(2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.cos(2* np.pi * x / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * (np.sin(2 * np.pi * x /L)**2)) / L
            
        def Fy1(x, y, c, p, A, L):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * x / L)) * np.sin(2 * np.pi * x / L))
        
        def Fx2(x, y, c, p, A, L):
            return -(5**(-p))*c*(y**p) * (-2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.sin(2* np.pi * x / L) \
                                          - 2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * x / L)) / L
                
        def Fy2(x, y, c, p, A, L):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * x / L)) * np.cos(2 * np.pi * x / L))
        
        if (sym == False):
            self.Fx = Fx1
            self.Fy = Fy1
        else:
            self.Fx = Fx2
            self.Fy = Fy2
        
        self.X = np.zeros((self.N, int(self.T/1000)))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, int(self.T/1000)))
        self.Y[:, 0] = self.y
        
        for i in range(1, self.T):
            
            
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            #self.dX[:, i] = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp1 = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += tmp1
            #self.x %= self.L
            #self.dY[:, i] = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp2 = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += tmp2
            #self.X[:, i] = self.x
            #self.Y[:, i] = self.y
            self.J[i] = np.mean(tmp1) / self.dt
            self.msd[i] = np.mean(np.square(self.x - np.mean(self.x)))
            
                #self.J[j] = np.mean(self.dX[:, i+1-10000:i+1]) / self.dt
                #print("instant total flux: %f" % self.J[j])
            if (i%1000 == 0):
                self.X[:, int(i/1000)] = self.x
                self.Y[:, int(i/1000)] = self.y
            
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            
    def anim(self, fold=True):
        fig, ax = plt.subplots(figsize=(5, 3))
        if (fold==True):
            ax.set(xlim=(0, 20), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'bo')[0]
            def animate(i):
                line.set_xdata(self.X[:, i]%self.L)
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('simple_fold_v0=%d.mp4'%(self.v0))
        else:
            ax.set(xlim=(-2000, 2020), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'ro')[0]
            def animate(i):
                line.set_xdata(self.X[:, i])
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('simple_v0=%d.mp4'%(self.v0))
        
            

class oscillate:
    
    def __init__(self, v0=1, D=0, Dr=0.1, dt=0.01, T=200000, N=1000, L=20, c=2, p=4, A=0.3, T_wall=10, sym=False, seed=0):
        np.random.seed(seed)
        
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
        self.T_wall = T_wall
        
        self.x = np.random.uniform(0, self.L, (self.N,))
        #self.x = np.zeros((self.N,))
        self.y = np.random.uniform(-5, 5, (self.N,))
        #self.y = np.zeros((self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        #self.X = np.zeros((self.N, self.T))
        #self.X[:, 0] = self.x
        #self.Y = np.zeros((self.N, self.T))
        #self.Y[:, 0] = self.y
        
        #self.dX = np.zeros((self.N, self.T))
        #self.dY = np.zeros((self.N, self.T))
        
        # total flux every 10000 iterations
        self.J = np.zeros(self.T)
        self.msd = np.zeros(self.T)
        
        # potential: V = c((y-0)/5)^p(1+A Exp[Cos[2 Pi x/L] ]Sin[2 Pi x/L])
        
        def Fx1(x, y, c, p, A, L, t, T_wall):
           return -(5**(-p))*c*(y**p)*(2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.cos(2* np.pi * x / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * (np.sin(2 * np.pi * x / L)**2)) / L * np.cos(2 * np.pi * t / T_wall)
            
        def Fy1(x, y, c, p, A, L, t, T_wall):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * x / L)) * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * t / T_wall))
        
        def Fx2(x, y, c, p, A, L, t, T_wall):
            return -(5**(-p))*c*(y**p)*(-2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.sin(2* np.pi * x / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * x / L)) * np.pi * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * x / L)) / L * np.cos(2 * np.pi * t / T_wall)
        def Fy2(x, y, c, p, A, L, t, T_wall):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * x / L)) * np.cos(2 * np.pi * x / L) * np.cos(2 * np.pi * t / T_wall))
        
        
        if (sym == False):
            self.Fx = Fx1
            self.Fy = Fy1
        else:
            self.Fx = Fx2
            self.Fy = Fy2
            
        self.X = np.zeros((self.N, int(self.T/100)))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, int(self.T/100)))
        self.Y[:, 0] = self.y
        
        for i in range(1, self.T):
            
            
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            #self.dX[:, i] = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp1 = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L, self.dt*i, self.T_wall) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += tmp1
            #self.x %= self.L
            #self.dY[:, i] = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp2 = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L, self.dt*i, self.T_wall) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += tmp2
            #self.X[:, i] = self.x
            #self.Y[:, i] = self.y
            self.J[i] = np.mean(tmp1) / self.dt
            self.msd[i] = np.mean(np.square(self.x - np.mean(self.x)))
            
                #self.J[j] = np.mean(self.dX[:, i+1-10000:i+1]) / self.dt
                #print("instant total flux: %f" % self.J[j])
            if (i%100 == 0):
                self.X[:, int(i/100)] = self.x
                self.Y[:, int(i/100)] = self.y
            
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            
    def anim(self, fold=True):
        fig, ax = plt.subplots(figsize=(5, 3))
        if (fold==True):
            ax.set(xlim=(0, 20), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'ro')[0]
            def animate(i):
                line.set_xdata(self.X[:, i]%self.L)
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('osc_sym_fold_v0=%d_Twall=%f_A=%f.mp4'%(self.v0, self.T_wall, self.A))
        else:
            ax.set(xlim=(-2000, 2020), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'ro')[0]
            def animate(i):
                line.set_xdata(self.X[:, i])
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('osc_v0=%d_Twall=%f_A=%f.mp4'%(self.v0, self.T_wall, self.A))

class move:
    
    def __init__(self, v0=1, D=0, Dr=0.1, dt=0.01, T=200000, N=1000, L=20, c=2, p=4, A=0.3, v_wall=1, sym=False, seed=0):
        np.random.seed(seed)
        
        # v0: constant velocity
        # s1: slope for first part
        # s2: (-s2) slope for second part
        # D: diffusion coefficient
        # Dr: rotational diffusion coefficient
        # dt: time interval
        # T: number of time steps
        # N: number of particles
        
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
        self.v_wall = v_wall
        
        self.x = np.random.uniform(0, self.L, (self.N,))
        #self.x = np.zeros((self.N,))
        self.y = np.random.uniform(-5, 5, (self.N,))
        #self.y = np.zeros((self.N,))
        self.theta = np.random.uniform(0, 2*np.pi, (self.N, ))
        
        
        self.J = np.zeros(self.T)
        self.msd = np.zeros(self.T)
        
        # asymmetric potential: V = c((y-0)/5)^p(1+A Exp[Cos[2 Pi x/L] ]Sin[2 Pi x/L])
        def Fx1(x, y, c, p, A, L, v, t):
           return -(5**(-p))*c*(y**p)*(2 * A * np.exp(np.cos(2 * np.pi * (x - v*t) / L)) * np.pi * np.cos(2* np.pi * (x - v*t) / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * (x - v*t) / L)) * np.pi * (np.sin(2 * np.pi * (x - v*t) / L)**2)) / L
            
        def Fy1(x, y, c, p, A, L, v, t):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * (x - v*t) / L)) * np.sin(2 * np.pi * (x - v*t) / L))
        
        # symmetric potential: V = c((y-0)/5)^p(1+A Exp[Cos[2 Pi x/L] ]Cos[2 Pi x/L])
        def Fx2(x, y, c, p, A, L, v, t):
           return -(5**(-p))*c*(y**p)*(-2 * A * np.exp(np.cos(2 * np.pi * (x - v*t) / L)) * np.pi * np.sin(2* np.pi * (x - v*t) / L) \
                                       - 2 * A * np.exp(np.cos(2 * np.pi * (x - v*t) / L)) * np.pi * np.cos(2 * np.pi * (x - v*t) / L) * np.sin(2 * np.pi * (x - v*t) / L)) / L
        def Fy2(x, y, c, p, A, L, v, t):
            return -(5**(-p))*c*p*(y**(p-1)) * (1 + A * np.exp(np.cos(2* np.pi * (x - v*t) / L)) * np.cos(2 * np.pi * (x - v*t) / L))
            
        if (sym == False):
            self.Fx = Fx1
            self.Fy = Fy1
        else:
            self.Fx = Fx2
            self.Fy = Fy2
        
        self.X = np.zeros((self.N, int(self.T/10)))
        self.X[:, 0] = self.x
        self.Y = np.zeros((self.N, int(self.T/10)))
        self.Y[:, 0] = self.y
        
        for i in range(1, self.T):
            
            
            #self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            #self.dX[:, i] = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp1 = self.Fx(self.x, self.y, self.c, self.p, self.A, self.L, self.v_wall, i*self.dt) * self.dt + v0 * np.cos(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.x += tmp1
            #self.x %= self.L
            #self.dY[:, i] = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            tmp2 = self.Fy(self.x, self.y, self.c, self.p, self.A, self.L, self.v_wall, i*self.dt) * self.dt + v0 * np.sin(self.theta) * self.dt + np.random.normal(0, np.sqrt(2 * D * self.dt), (N, ))
            self.y += tmp2
            #self.X[:, i] = self.x
            #self.Y[:, i] = self.y
            self.J[i] = np.mean(tmp1) / self.dt
            self.msd[i] = np.mean(np.square(self.x - np.mean(self.x)))
            
                #self.J[j] = np.mean(self.dX[:, i+1-10000:i+1]) / self.dt
                #print("instant total flux: %f" % self.J[j])
            if (i%10 == 0):
                self.X[:, int(i/10)] = self.x
                self.Y[:, int(i/10)] = self.y
            
            self.theta += np.random.normal(0, np.sqrt(2 * Dr * self.dt), (self.N, ))
            
    def anim(self, fold=True):
        fig, ax = plt.subplots(figsize=(5, 3))
        if (fold==True):
            ax.set(xlim=(0, 20), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'bo')[0]
            def animate(i):
                line.set_xdata(self.X[:, i]%self.L)
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('move_fold_v0=%d_vwall=%d.mp4'%(self.v0, self.v_wall))
        else:
            ax.set(xlim=(-2000, 2020), ylim=(-20, 20))
            line = ax.plot(self.X[:, 0], self.Y[:, 0], 'ro')[0]
            def animate(i):
                line.set_xdata(self.X[:, i])
                line.set_ydata(self.Y[:, i])
            anim = FuncAnimation(fig, animate, interval=100, frames=self.X.shape[1])
            anim.save('move_v0=%d_vwall=%d.mp4'%(self.v0, self.v_wall))


def plot_traj(x, y, c='r'):
    f = plt.figure(figsize=(5,3))
    plt.plot(x, y, '%so'%(c), alpha=0.2)
    plt.title("a single trajectory")

def unfold_X(dX):
    return dX.cumsum(axis=1)

def rel_msd(X, v, t):
    return np.square(X - v*t).mean(axis=0)

def MSD(X):
    return np.square(X).mean(axis=0)

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