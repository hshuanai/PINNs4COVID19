# -*- coding: utf-8 -*-

import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy import integrate

# cd dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import constants
import numpy as np
import pandas as pd


# load data
path, paras = constants.read_data_with_timespan('Germany',['2021-02-23','2021-07-01'])
pf = pd.read_csv(path)

country = paras[0]
N = int(paras[1])
days =int(paras[-1]) 
date = np.array(pf['date'])

# normalized
# Ic_raw = np.array(pf['Ic'])/N
# Inew_raw = np.array(pf['Inew'])/N
# Recovered_raw = np.array(pf['R'])/N
# D_raw = np.array(pf['D'])/N
# N = N/N

# calculate S,I,R
I_raw = np.array(pf['I_4_SIR'])/N*30
R_raw = np.array(pf['R_4_SIR'])/N*15
N = N/N
S_raw = N-I_raw-R_raw

top = int(np.argmax(I_raw))

# training /test set
train_size = (top-10)
test_size = -train_size

device = constants.get_device()

# SIR model 
def covid_sir(u, t, beta, gamma):
    S, I, R= u
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I

    return np.array([dS_dt, dI_dt, dR_dt])


# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.001)


# SIR NN
class Net_SIR(nn.Module):
    def __init__(self, layers=None,):
        super(Net_SIR, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,64),nn.Tanh(), # 1 means time
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,3),nn.Sigmoid() # values of each compartments relative to SIR 
        )

    def forward(self, x):
        output = self.layers(x)
        return output

    def set_layer(self, layers):
        self.layers = layers


pinn_sir = Net_SIR()
pinn_sir = pinn_sir.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)*N


# LOSS = data loss + residual loss
def residual_loss(t,beta,gamma):
    # day unit is 1
    dt = 1
    u = network(t)

    st2, st1 = u[1:,0], u[:-1,0]
    it2, it1 = u[1:,1], u[:-1,1]
    rt2, rt1 = u[1:,2], u[:-1,2]

    ds = (st2 - st1)/dt
    di = (it2 - it1)/dt
    dr = (rt2 - rt1)/dt

    loss_s = (-beta*st1*it1-ds)**2
    loss_i = (beta*st1*it1-gamma*it1-di)**2
    loss_r = (gamma*it1-dr)**2
    loss_n = (u[:,0]+u[:,1]+u[:,2]-N)**2 # normalization constraint

    return loss_s.sum(), loss_i.sum(), loss_r.sum(), loss_n.sum()

  
def init_loss(t,S0,I0,R0):
    u = network(t)

    loss_s0 = (u[0,0]- S0)**2
    loss_i0 = (u[0,1]- I0)**2
    loss_r0 = (u[0,2]- R0)**2

    return loss_s0+ loss_i0+ loss_r0

def data_loss(t,index):
    u = network(t)

    loss_st = (u[index,0] - torch.from_numpy(S_raw[index]).to(device))**2
    loss_it = (u[index,1] - torch.from_numpy(I_raw[index]).to(device))**2
    loss_rt = (u[index,2] - torch.from_numpy(R_raw[index]).to(device))**2

    return loss_st.sum(), loss_it.sum(), loss_rt.sum()

# initial parameters
beta_raw = 0.25 
gamma_raw = 0.15

optimizer = optim.Adam(pinn_sir.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

beta_trained = Variable(torch.tensor([beta_raw]).to(device), requires_grad=True)
gamma_trained = Variable(torch.tensor([gamma_raw]).to(device), requires_grad=True)

optimizer_v = optim.Adam([beta_trained, gamma_trained], lr=1e-4, weight_decay=0.01)
scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=5000, gamma=0.998)

S0 = S_raw[0]
I0 = I_raw[0]
R0 = R_raw[0]

# time points(discrete)
t_points = np.linspace(0,days,days+1)[:-1]
# time points(training set)shuffle
index = torch.randperm(train_size)

total_epoch = []
total_data_loss_epoch = []
total_residual_loss_epoch = []

early_stopping = 500
# two head queue
loss_history = deque(maxlen=early_stopping + 1)

alpha_0 = 1000
alpha_1 = 1000000
alpha_2 = 500

pinn_sir.zero_grad()

for epoch in range(10000000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()
    optimizer_v.zero_grad()

    loss_s,loss_i,loss_r,loss_n = residual_loss(t_points,beta_trained,gamma_trained)
    loss_st, loss_it, loss_rt = data_loss(t_points, index)
    loss_init = init_loss(t_points,S0,I0,R0)

    loss = alpha_0*(loss_st+ 800*loss_it+ 300*loss_rt)+ alpha_1*(loss_s+ loss_i+ loss_r+ loss_n)+ alpha_2*loss_init

    running_loss += loss.item()
    loss_history.append(running_loss)
    total_epoch.append(running_loss)
    
    total_data_loss_epoch.append((loss_st.item()+loss_it.item()+loss_rt.item()))
    total_residual_loss_epoch.append((loss_s+loss_i+loss_r+loss_n).item())

    # 如果队列满了并且弹出的第一个值小于队列中剩余的最小值，表明该值为最小值，在队列长度个epoch内loss不再下降
    if len(loss_history) > early_stopping and loss_history.popleft() < min(loss_history):
      print(f"Early stopping at [{epoch}] times. No train loss improvement in [{early_stopping}] epochs.")
      break
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    optimizer_v.step()
    scheduler_v.step()
    
    if (epoch % 1000 == 0):   
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: {(loss_st.item()+loss_it.item()+loss_rt.item())}, S loss: {loss_st.item()}, I loss: {loss_it.item()}, R loss: {loss_rt.item()}.')
        print(f'residual loss: {(loss_s+loss_i+loss_r+loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init}.')

print('Finished Training')

# plot the loss
constants.plot_log_loss(country,total_epoch,total_data_loss_epoch,total_residual_loss_epoch,train_size)

u = network(t_points)
st = u[:,0].cpu().detach().numpy()
it = u[:,1].cpu().detach().numpy()
rt = u[:,2].cpu().detach().numpy()

beta, gamma = beta_trained.detach().numpy()[0], gamma_trained.detach().numpy()[0]

# solves the SIR_ODEs by parameters learned fromm PINNs
u0 = [S0, I0, R0]
res = integrate.odeint(covid_sir, u0, t_points, args=(beta,gamma))
S_ode, I_ode, R_ode = res.T

constants.plot_result_comparation(country,st,'S',S_raw,S_ode,train_size)
constants.plot_result_comparation(country,it,'I',I_raw,I_ode,train_size)
constants.plot_result_comparation(country,rt,'R',R_raw,R_ode,train_size)

# save results
constants.save_results_fixed(country,date,st,it,rt,S_raw,I_raw,R_raw,S_ode,I_ode,R_ode,train_size)

# save MSE and MAE
constants.save_error_result_sir(country,S_raw,I_raw,R_raw,st,it,rt,S_ode,I_ode,R_ode,train_size)

# save parameters learned from PINNs
constants.save_parameters_learned(country,beta_raw,gamma_raw,beta,gamma,train_size)

# print beta and gamma
print(f'days is {days}, peak index is: {top}.')
print(f'alpha_0: {alpha_0}, alpha_1: {alpha_1}, alpha_2: {alpha_2}.')
print(f'beta_raw: {beta_raw}, gamma_raw: {gamma_raw}.')
print(f'beta and gamma are: {beta}, {gamma}.')
