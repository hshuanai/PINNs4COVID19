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

# 将上两级目录加入到系统路径中（constants.py所在目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import constants
import numpy as np
import pandas as pd

# loading dataset
path, paras = constants.read_data('Germany')
pf = pd.read_csv(path)

country = paras[0]
N = int(paras[1])
days =int(paras[-1]) 
date = np.array(pf['date'])

# data over N
Ic_raw = np.array(pf['Ic'])/N
Inew_raw = np.array(pf['Inew'])/N
Recovered_raw = np.array(pf['R'])/N
D_raw = np.array(pf['D'])/N
N = N/N

top = int(np.argmax(Inew_raw)) 

# 训练集/测试集
train_size = top-5
test_size = -days

device = constants.get_device()

# SIR model 输出为单位时间内的每个舱室的个体数
def covid_sir(u, t, beta, gamma):
    S, I, R, Ic= u
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I
    dIc_dt = beta*S*I

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
            nn.Linear(1,32),nn.Tanh(), # 1 means time
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,3),nn.Sigmoid() # [I,R,Ic] values of each compartments relative to SIR, and Ic
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


# 初始化 NNs
pinn_sir = Net_SIR()
pinn_sir.apply(init_weights)
pinn_sir = pinn_sir.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)


t=np.linspace(0,days,days+1)[:-1]
SS,II,RR,ICC,Inewnew = network(t)

constants.plot_show_results(country,'S',SS.detach().numpy(),SS.detach().numpy(),'show')

constants.plot_show_results(country,'I',II.detach().numpy(),II.detach().numpy(),'show')

constants.plot_show_results(country,'R',RR.detach().numpy(),RR.detach().numpy(),'show')

constants.plot_show_results(country,'Ic',ICC.detach().numpy(),ICC.detach().numpy(),'show')

constants.plot_show_results(country,'Inew',Inewnew.detach().numpy(),Inewnew.detach().numpy(),'show')



# LOSS = data loss + residual loss
def residual_loss(S,I,R,Ic):
    dt = 1

    beta = beta_f()
    gamma = gamma_f()

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

    return loss_s.sum(), loss_i.sum(), loss_r.sum(), loss_ic.sum(), loss_n.sum(), beta.item(), gamma.item()


def init_loss(S,I,R):

    I0 = I0_f()
    R0 = Ic_raw[0]- I0
    S0 = N- Ic_raw[0]

    loss_s0 = (S[0] - S0)**2
    loss_i0 = (I[0] - I0)**2
    loss_r0 = (R[0] - R0)**2
    loss_N = (S0+ I0+ R0- N)**2

    return loss_s0+ loss_i0+ loss_r0+loss_N, S0.item(), I0.item(), R0.item()


def data_loss(Ic,Inew,index):

    loss_ic = (Ic[index] - torch.from_numpy(Ic_raw[:-1][index]).to(device))**2
    loss_inew = (Inew[index] - torch.from_numpy(Inew_raw[index]).to(device))**2

    return loss_ic.sum(), loss_inew.sum()

beta_init,gamma_init,I0_init = 0.2,0.5,0.05
R0_init = Ic_raw[0]-I0_init
S0_init = N- Ic_raw[0]

beta_raw,gamma_raw,I0_raw = torch.tensor(beta_init),torch.tensor(gamma_init),torch.tensor(I0_init)
# beta_raw, gamma_raw, I0_raw = torch.rand(1), torch.rand(1), torch.rand(1)
print(f'beta_raw: {beta_raw}, gamma_raw: {gamma_raw}, I0_raw: {I0_raw}.')

beta_trained = nn.Parameter(beta_raw,requires_grad=True)
gamma_trained = nn.Parameter(gamma_raw,requires_grad=True)
I0_trained = nn.Parameter(I0_raw,requires_grad=True)


def beta_f():
    # return 0.05+(0.5-0.05)*torch.sigmoid(beta_trained)
    return torch.sigmoid(beta_trained)
    

def gamma_f():
    return torch.sigmoid(gamma_trained)
    # return 0.05+(0.6-0.05)*torch.sigmoid(gamma_trained)


def I0_f():
    return torch.sigmoid(I0_trained)


# nn_parameters = list(pinn_sir.parameters())
# nn_parameters.extend([beta_trained, gamma_trained, I0_trained])

optimizer = optim.Adam(pinn_sir.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

optimizer_paras = optim.Adam([beta_trained, gamma_trained, I0_trained], lr=1e-4, weight_decay=0.01)
scheduler_paras = optim.lr_scheduler.StepLR(optimizer_paras, step_size=5000, gamma=0.998)

# 训练集离散时间点
t_train = np.linspace(0,train_size,train_size+1)[:-1]
# 训练集时间点shuffle
index = torch.randperm(train_size-1)

total_epoch_loss,total_data_loss,total_residuals_loss = [],[],[]
early_stopping = 2000
#双头队列
loss_history = deque(maxlen=early_stopping+ 1)

alpha_0 = 1000
alpha_1 = 0

pinn_sir.zero_grad()


beta_final,gamma_final,S0_final,I0_final,R0_final = 0.0,0.0,0.0,0.0,0.0

for epoch in range(1000000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()
    optimizer_paras.zero_grad()

    S,I,R,Ic,Inew = network(t_train)
    loss_s,loss_i,loss_r,loss_ic,loss_n, beta_final,gamma_final = residual_loss(S,I,R,Ic)
    ic_loss, loss_inew = data_loss(Ic,Inew,index)
    loss_init,S0_final,I0_final,R0_final = init_loss(S,I,R)

    loss = alpha_0*(0*ic_loss+ 500*3*loss_inew)+ alpha_1*(loss_s+ loss_i+ loss_r+ loss_ic+ loss_n+ loss_init)

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

    optimizer_paras.step()
    scheduler_paras.step()
    
    if (epoch % 1000 == 0):   
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: {(loss_ic.item()+loss_inew.item())}, Ic loss: {ic_loss.item()}, Inew loss: {loss_inew.item()}.')
        print(f'residual loss: {(loss_s+loss_i+loss_r+loss_ic+loss_n).item()}, loss_s: {loss_s.item()}, loss_i: {loss_i.item()}, loss_r: {loss_r.item()}, loss_ic: {loss_ic.item()}, loss_n: {loss_n.item()}.')
        print(f'init loss: {loss_init.item()}.')
        print(f'beta: {beta_final}, gamma: {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Finished Training, beta: {beta_final}, gamma: {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')


# 绘制loss图
constants.plot_log_loss(country,total_epoch_loss,total_data_loss,total_residuals_loss,'fixed')

t = np.linspace(0,days,days+1)[:-1]

St,It,Rt,Ict,Inewt = network(t)
st = St.cpu().detach().numpy()
it = It.cpu().detach().numpy()
rt = Rt.cpu().detach().numpy()
ict = Ict.cpu().detach().numpy()
inewt = Inewt.cpu().detach().numpy()


# 利用训练参数求解ODE SIR
u0 = [S0_final,I0_final,R0_final,Ic_raw[0]]
res = integrate.odeint(covid_sir, u0, t, args=(beta_final,gamma_final))
S_ode, I_ode, R_ode, Ic_ode = res.T
Inew_ode = Ic_ode[1:]- Ic_ode[:-1]

# S,I,R
constants.plot_results(country,'S',st,S_ode,'fixed')
constants.plot_results(country,'I',it,I_ode,'fixed')
constants.plot_results(country,'R',rt,R_ode,'fixed')
# Ic,Inew
constants.plot_results_comparation(country,'Ic', Ic_raw, ict, Ic_ode, train_size,'fixed')
constants.plot_results_comparation(country,'Inew',Inew_raw[:-1], inewt, Inew_ode, train_size,'fixed')

# 保存结果
# constants.save_results_fixed_parameters(country,date,st,it,rt,ict,np.append(inewt,inewt[-1]),Ic_raw,Inew_raw[:-1],S_ode,I_ode,R_ode,Ic_ode)

# 保存parameters learned. [S0,I0,R0,beta,gamma]
constants.save_fixed_parameters_result(country,beta_init,gamma_init,S0_init,I0_init,R0_init,beta_final,gamma_final,S0_final,I0_final,R0_final)


# 打印beta 和 gamma
print(f'peak index is: {top}.')
print(f'beta and gamma is: {beta_final}, {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')