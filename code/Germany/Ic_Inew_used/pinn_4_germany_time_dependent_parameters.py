# -*- coding: utf-8 -*-

import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate
from scipy.interpolate import interp1d

# 将上两级目录加入到系统路径中（constants.py所在目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import constants
import numpy as np
import pandas as pd

# 加载数据
path, paras = constants.read_data('Germany')
pf = pd.read_csv(path)

country = paras[0]
N = int(paras[1])
days =int(paras[-1]) 
date = np.array(pf['date'])

# 归一化，便于计算
Ic_raw = np.array(pf['Ic'])/N
Inew_raw = np.array(pf['Inew'])/N
Recovered_raw = np.array(pf['R'])/N
D_raw = np.array(pf['D'])/N
N = N/N

top = int(np.argmax(Inew_raw)) 

# 训练集/测试集
train_size = days
test_size = -days

device = constants.get_device()


# SIR model 输出为单位时间内的每个舱室的个体数
def covid_sir(u, t, beta, gamma):
    S, I, R, Ic= u

    dS_dt = -beta_func(t)*S*I
    dI_dt = beta_func(t)*S*I- gamma_func(t)*I
    dR_dt = gamma_func(t)*I
    dIc_dt = beta_func(t)*S*I

    return np.array([dS_dt, dI_dt, dR_dt, dIc_dt])


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)


# SIR NN, NN的输入为t，输出为每个舱室对应t的值，t->S,I,R,Ic
class Net_SIR(nn.Module):
    def __init__(self):
        super(Net_SIR, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,64),nn.Tanh(), # 1 means time
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,3,nn.Sigmoid()) # [I,R,Ic] values of each compartments relative to SIR, and Ic
        )

    def forward(self, x):
        output = self.layers(x)
        I = output[:,0]
        R = output[:,1]
        Ic = output[:,2]
        S = N- I- R
        Inew = Ic[1:]- Ic[:-1]
        return S,I,R,Ic,Inew

    def set_layer(self, layers):
        self.layers = layers


class Net_beta(nn.Module):
    def __init__(self):
        super(Net_beta, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,32),nn.Tanh(), # 1 means time
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,1),nn.Sigmoid() # force into [0,1]
        )

    def forward(self, x):
        output = self.layers(x)
        return output

    def set_layer(self, layers):
        self.layers = layers


class Net_gamma(nn.Module):
    def __init__(self):
        super(Net_gamma, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,32),nn.Tanh(), # 1 means time
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,1),nn.Sigmoid() # force into [0,1]
        )

    def forward(self, x):
        output = self.layers(x)
        return output

    def set_layer(self, layers):
        self.layers = layers


# 初始化 NNs
pinn_sir = Net_SIR()
pinn_sir.apply(init_weights)
pinn_sir = pinn_sir.to(device)

nn_beta = Net_beta()
nn_beta.apply(init_weights)
nn_beta = nn_beta.to(device)

nn_gamma = Net_gamma()
nn_gamma.apply(init_weights)
nn_gamma = nn_gamma.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)


def network_beta(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return nn_beta(input)


def network_gamma(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return nn_gamma(input)


# LOSS = data loss + residual loss
def residual_loss(S,I,R,Ic,beta,gamma):
    dt = 1

    st2, st1 = S[1:], S[:-1]
    it2, it1 = I[1:], I[:-1]
    rt2, rt1 = R[1:], R[:-1]
    ict2, ict1 = Ic[1:], Ic[:-1]

    ds = (st2 - st1)/dt
    di = (it2 - it1)/dt
    dr = (rt2 - rt1)/dt
    dic = (ict2 - ict1)/dt

    loss_s = (-beta*st1*it1-ds)**2
    loss_i = (beta*st1*it1-gamma*it1-di)**2
    loss_r = (gamma*it1-dr)**2
    loss_ic = (beta*st1*it1-dic)**2
    loss_n = (S+ I+ R- N)**2 # normalization constraint

    return loss_s.sum(), loss_i.sum(), loss_r.sum(), loss_ic.sum(), loss_n.sum()


def init_loss(S,I,R):

    I0 = I0_f()
    R0 = Ic_raw[0]- I0
    S0 = N- Ic_raw[0]

    loss_s0 = (S[0] - S0)**2
    loss_i0 = (I[0] - I0)**2
    loss_r0 = (R[0] - R0)**2

    return loss_s0+ loss_i0+ loss_r0, S0.item(), I0.item(), R0.item()


def data_loss(Ic,Inew,index):

    loss_ic = (Ic[index] - torch.from_numpy(Ic_raw[:-1][index]).to(device))**2
    loss_inew = (Inew[index] - torch.from_numpy(Inew_raw[index]).to(device))**2

    return loss_ic.sum(), loss_inew.sum()

I0_init = 0.05
R0_init = Ic_raw[0]-I0_init
S0_init = N- Ic_raw[0]

print(f'I0_raw: {I0_init}.')
I0_trained = nn.Parameter(torch.tensor(I0_init),requires_grad=True)


def I0_f():
    return torch.sigmoid(I0_trained)


nn_parameters = list(pinn_sir.parameters())
# nn_beta_parameters = list(nn_beta.parameters())
# nn_gamma_parameters = list(nn_gamma.parameters())

nn_parameters.extend([I0_trained])
# nn_parameters.extend(nn_beta_parameters)
# nn_parameters.extend(nn_gamma_parameters)

# optimizer = optim.Adam(nn_parameters, lr=1e-4, weight_decay=0.01)

optimizer = optim.AdamW([{'params':nn_parameters, 'lr':1e-4,'weight_decay':0.01}
                        ,{'params':nn_beta.parameters(), 'lr':1e-4,'weight_decay':0.01}
                        ,{'params':nn_gamma.parameters(), 'lr':1e-4,'weight_decay':0.01}])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

# optimizer_beta = optim.Adam(nn_beta.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler_beta  = optim.lr_scheduler.StepLR(optimizer_beta, step_size=5000, gamma=0.998)

# optimizer_gamma = optim.Adam(nn_gamma.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler_gamma = optim.lr_scheduler.StepLR(optimizer_gamma, step_size=5000, gamma=0.998)


# 训练集离散时间点
t_train = np.linspace(0,train_size,train_size+1)[:-1]
# 训练集时间点shuffle
index = torch.randperm(train_size-1)

total_epoch_loss,total_data_loss,total_residuals_loss = [],[],[]
early_stopping = 5000
#双头队列
loss_history = deque(maxlen=early_stopping+ 1)

alpha_0 = 500
alpha_1 = 20

pinn_sir.zero_grad()
nn_beta.zero_grad()
nn_gamma.zero_grad()

S0_final,I0_final,R0_final = 0.0,0.0,0.0

for epoch in range(500000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()
    # optimizer_beta.zero_grad()
    # optimizer_gamma.zero_grad()

    S,I,R,Ic,Inew = network(t_train)
    beta = network_beta(t_train)
    gamma = network_gamma(t_train)

    loss_s,loss_i,loss_r,loss_ic,loss_n = residual_loss(S,I,R,Ic,beta,gamma)
    ic_loss, loss_inew = data_loss(Ic,Inew,index)
    loss_init,S0_final,I0_final,R0_final = init_loss(S,I,R)

    loss = alpha_0*(120*2*ic_loss+ 120*50*loss_inew)+ alpha_1*(loss_s+ loss_i+ loss_r+ loss_ic+ loss_n+ loss_init)

    running_loss += loss.item()
    loss_history.append(running_loss)
    total_epoch_loss.append(running_loss)
    total_data_loss.append(loss_ic.item()+loss_inew.item())
    total_residuals_loss.append((loss_s+loss_i+loss_r+loss_ic+loss_n+loss_init).item())

    # 如果队列满了并且弹出的第一个值小于队列中剩余的最小值，表明该值为最小值，在队列长度个epoch内loss不再下降
    if len(loss_history) > early_stopping and loss_history.popleft() < min(loss_history):
      print(f"Early stopping at [{epoch}] times. No train loss improvement in [{early_stopping}] epochs.")
      break
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    # optimizer_beta.step()
    # scheduler_beta.step()
    # optimizer_gamma.step()
    # scheduler_gamma.step()
    
    if (epoch % 1000 == 0):   
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: {(loss_ic.item()+loss_inew.item())}, Ic loss: {ic_loss.item()}, Inew loss: {loss_inew.item()}.')
        print(f'residual loss: {(loss_s+loss_i+loss_r+loss_ic+loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_ic: {loss_ic.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init.item()}.')
        print(f'S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Finished Training, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')


# 绘制loss图
constants.plot_log_loss(country,total_epoch_loss,total_data_loss,total_residuals_loss,'time_dependent')

t = np.linspace(0,days,days+1)[:-1]

St,It,Rt,Ict,Inewt = network(t)
st = St.cpu().detach().numpy()
it = It.cpu().detach().numpy()
rt = Rt.cpu().detach().numpy()
ict = Ict.cpu().detach().numpy()
inewt = Inewt.cpu().detach().numpy()

beta = network_beta(t).cpu().detach().numpy().flatten()
gamma = network_gamma(t).cpu().detach().numpy().flatten()


sample_times = np.arange(len(t))
beta_func = interp1d(sample_times, beta, bounds_error=False, fill_value="extrapolate")
gamma_func = interp1d(sample_times, gamma, bounds_error=False, fill_value="extrapolate")


# 利用训练参数求解ODE SIR
u0 = [S0_final,I0_final,R0_final,Ic_raw[0]]
res = integrate.odeint(covid_sir, u0, t, args=(beta,gamma))
S_ode, I_ode, R_ode, Ic_ode = res.T
Inew_ode = Ic_ode[1:]-Ic_ode[:-1]

# beta, gamma
constants.plot_parameters_results(country,'beta',beta)
constants.plot_parameters_results(country,'gamma',gamma)

# S,I,R
constants.plot_results(country,'S',st,S_ode,'time_dependent')
constants.plot_results(country,'I',it,I_ode,'time_dependent')
constants.plot_results(country,'R',rt,R_ode,'time_dependent')
# Ic,Inew
constants.plot_results_comparation(country,'Ic', Ic_raw, ict, Ic_ode, train_size,'time_dependent')
constants.plot_results_comparation(country,'Inew',Inew_raw[:-1], inewt, Inew_ode, train_size,'time_dependent')

# 保存结果
# constants.save_results_time_dependent_parameters(country,date,st,it,rt,ict,np.append(inewt,inewt[-1]),Ic_raw,Inew_raw[:-1],S_ode,I_ode,R_ode,Ic_ode,beta,gamma)

# 保存parameters learned. [S0,I0,R0,beta,gamma]
constants.save_time_dependent_parameters_result(country,S0_init,I0_init,R0_init,S0_final,I0_final,R0_final)


# 打印beta 和 gamma
print(f'peak index is: {top}.')
print(f'S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')


