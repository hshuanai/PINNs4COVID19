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
Ic_raw = np.array(pf['Ic'])/N
Inew_raw = np.array(pf['Inew'])/N
Recovered_raw = np.array(pf['R'])/N
D_raw = np.array(pf['D'])/N
N = N/N

# training /test set
train_size = days

device = constants.get_device()

# SIR model 
def covid_sir(u, t, beta, gamma):
    S, I, R= u
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I

    return np.array([dS_dt, dI_dt, dR_dt])


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
        I = output[:,0]
        R = output[:,1]
        Ic = output[:,2]
        S = N-I-R
        return S,I,R,Ic

    def set_layer(self, layers):
        self.layers = layers


pinn_sir = Net_SIR()
pinn_sir = pinn_sir.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)


# LOSS = data loss + residual loss
def residual_loss(us,ui,ur,uic):
    # day unit is 1
    dt = 1
    beta,gamma = beta_f(),gamma_f()

    st2, st1 = us[1:], us[:-1]
    it2, it1 = ui[1:], ui[:-1]
    rt2, rt1 = ur[1:], ur[:-1]
    ict2, ict1 = uic[1:], uic[:-1]

    ds = (st2 - st1)/dt
    di = (it2 - it1)/dt
    dr = (rt2 - rt1)/dt
    dic = (ict2 - ict1)/dt

    loss_s = (-beta*st1*it1-ds)**2
    loss_i = (beta*st1*it1-gamma*it1-di)**2
    loss_r = (gamma*it1-dr)**2
    loss_ic = (beta*st1*it1-dic)**2

    loss_n = (us+ui+ur-N)**2 # normalization constraint

    return loss_s.sum(), loss_i.sum(), loss_r.sum(),loss_ic.sum(), loss_n.sum(), beta.item(), gamma.item()

  
def init_loss(us,ui,ur):

    I0 = I0_f()
    R0 = Ic_raw[0]- I0
    S0 = N- Ic_raw[0]

    loss_s0 = (us[0]- S0)**2
    loss_i0 = (ui[1]- I0)**2
    loss_r0 = (ur[2]- R0)**2

    return loss_s0+ loss_i0+ loss_r0, S0.item(), I0.item(), R0.item()

def data_loss(uic,index):

    loss_ict = (uic - torch.from_numpy(Ic_raw[index]).to(device))**2
    return loss_ict.sum()


def beta_f():
    return torch.sigmoid(beta_trained)
    

def gamma_f():
    return torch.sigmoid(gamma_trained)


def I0_f():
    return torch.sigmoid(i0_trained)


# initial parameters
beta_raw = 0.25 
gamma_raw = 0.15
i0_raw = 0.05

optimizer = optim.Adam(pinn_sir.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

beta_trained = Variable(torch.tensor([beta_raw]).to(device), requires_grad=True)
gamma_trained = Variable(torch.tensor([gamma_raw]).to(device), requires_grad=True)
i0_trained = Variable(torch.tensor([i0_raw]).to(device), requires_grad=True)

optimizer_v = optim.Adam([beta_trained, gamma_trained,i0_trained], lr=1e-4, weight_decay=0.01)
scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=5000, gamma=0.998)

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

alpha_0 = 500
alpha_1 = 2000
alpha_2 = 200

pinn_sir.zero_grad()
beta_final,gamma_final,S0_final,I0_final,R0_final = 0.0,0.0,0.0,0.0,0.0
for epoch in range(500000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()
    optimizer_v.zero_grad()

    us,ui,ur,uic = network(t_points)
    loss_s,loss_i,loss_r,loss_ic, loss_n, beta_final,gamma_final = residual_loss(us,ui,ur,uic)
    loss_ict = data_loss(ur, index)
    loss_init,S0_final,I0_final,R0_final = init_loss(us,ui,ur)

    loss = alpha_0*(50*loss_ict)+ alpha_1*(loss_s+ loss_i+ loss_r+ loss_ic+ loss_n)+ alpha_2*loss_init

    running_loss += loss.item()
    loss_history.append(running_loss)
    total_epoch.append(running_loss)
    
    total_data_loss_epoch.append(loss_ict.item())
    total_residual_loss_epoch.append((loss_s+loss_i+loss_r+loss_ic+loss_n).item())

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
        print(f'data loss: {(loss_ict.item())}.')
        print(f'residual loss: {(loss_s+loss_i+loss_r+loss_ic+loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_ic: {loss_ic.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init.item()}.')
        print(f'beta: {beta_final}, gamma: {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Finished Training, beta: {beta_final}, gamma: {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')



# plot the loss
constants.plot_log_loss(country,total_epoch,total_data_loss_epoch,total_residual_loss_epoch,train_size)

st,it,rt,ict = network(t_points)
S_pinns = st.cpu().detach().numpy()
I_pinns = it.cpu().detach().numpy()
R_pinns = rt.cpu().detach().numpy()
Ic_pinns = ict.cpu().detach().numpy()

# solves the SIR_ODEs by parameters learned fromm PINNs
u0 = [S0_final, I0_final, R0_final]
res = integrate.odeint(covid_sir, u0, t_points, args=(beta_final,gamma_final))
S_ode, I_ode, R_ode = res.T

constants.plot_result_comparation(country,S_pinns,'S',S_ode,S_ode,train_size)
constants.plot_result_comparation(country,I_pinns,'I',I_ode,I_ode,train_size)
constants.plot_result_comparation(country,R_pinns,'R',R_ode,R_ode,train_size)
constants.plot_result_comparation(country,Ic_pinns,'Ic',Ic_pinns,Ic_pinns,train_size)

# save results
# constants.save_results_fixed(country,date,S_pinns,I_pinns,R_pinns,S_raw,I_raw,R_raw,S_ode,I_ode,R_ode,train_size)

# save MSE and MAE
# constants.save_error_result_sir(country,S_raw,I_raw,R_raw,st,it,rt,S_ode,I_ode,R_ode,train_size)

# save parameters learned from PINNs
constants.save_parameters_learned(country,beta_raw,gamma_raw,beta_final,gamma_final,train_size)

# print beta and gamma
print(f'days is {days}.')
print(f'alpha_0: {alpha_0}, alpha_1: {alpha_1}, alpha_2: {alpha_2}.')
print(f'beta_raw: {beta_raw}, gamma_raw: {gamma_raw}.')
print(f'beta and gamma I0 are: {beta_final}, {gamma_final}, {I0_final}.')
