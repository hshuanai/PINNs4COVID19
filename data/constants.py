# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn import metrics

path_confirmed = './dataset_raw/time_series_covid19_confirmed_global.csv'
path_deaths = './dataset_raw/time_series_covid19_deaths_global.csv'
path_recovered = './dataset_raw/time_series_covid19_recovered_global.csv'

path_plots_dataprocessed = './plots_dataset/'
path_results = './results/'
path_dataprocessed = './dataset_processed/'


countries = ['China']#,'Germany','Italy','Greece','Switzerland','Spain','Sweden']
# 'Germany','Italy','Greece','Switzerland','Spain','Sweden'
populations = [1411778724]#84323763,60627498,10329287,8909885,46472245,10306294] 
timespans = {'China':['2022-01-01','2023-01-01']}
                # ,'Germany':['2021-02-23','2021-07-01']
                # ,'Italy':['2021-02-23','2021-07-01']
                # ,'Greece':['2021-02-23','2021-07-01']
                # ,'Switzerland':['2021-02-23','2021-07-01']
                # ,'Spain':['2021-02-23','2021-07-01']
                # ,'Sweden':['2021-02-23','2021-07-01']}


def read_data(country_name):
    for country in os.listdir(path_dataprocessed):
        if country_name in country:
            paras = country.split('_')[:-1]
            return path_dataprocessed+country,paras
    return f'Not support {country_name}.'


def read_data_with_timespan(country_name,timespan):
    for country in os.listdir(path_dataprocessed):
        paras = country.split('_')[:-1]
        if country_name == paras[0] and timespan[0] == paras[2] and timespan[1] == paras[3]:
            return path_dataprocessed+country,paras
    return f'Not support {country_name}.'


def get_device():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if device.type == 'cuda':
        print('Using device:', torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        torch.cuda.set_device(0)
    return device


def plots(country, data, name, data_type):
    plt.figure(figsize=(16, 9))
    t = np.linspace(0, len(data), len(data)+1)[:-1]

    # 真实值
    plt.plot(t, data, color='black', label=name, linewidth=2)

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.legend(fontsize=25)
    plt.savefig(path_plots_dataprocessed +
                f'{country}_{name}_{data_type}_realdata.pdf', dpi=600)
    plt.close()


def plot_old_loss(country,loss,):
    plt.figure(figsize=(16, 9))
    eps = np.linspace(0, len(loss), len(loss))

    plt.plot(eps, np.log10(np.array(loss)), linewidth=1, label='loss')

    plt.title('train loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend()

    # 设置刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 设置图例字体大小
    plt.legend(loc=7, fontsize=16)

    plt.savefig(path_results+f'{country}_loss.pdf',dpi=600)
    plt.close()


def plot_loss(country,loss,type):
    plt.figure(figsize=(16, 9))
    eps = np.linspace(0, len(loss), len(loss))

    plt.plot(eps, np.log10(np.array(loss)), linewidth=1, label='loss')
    # plt.plot(eps, np.log10(data_loss), linewidth=1, label='data_loss')
    # plt.plot(eps, np.log10(residuals_loss), linewidth=1, label='residuals_loss')

    plt.title('train loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xlabel('log10', fontsize=20)
    plt.legend()

    # 设置刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 设置图例字体大小
    plt.legend(loc=7, fontsize=16)

    plt.savefig(path_results+f'{country}_{type}_loss.pdf',dpi=600)
    plt.close()


def plot_log_loss(country,loss,data_loss,residuals_loss,type):
    plt.figure(figsize=(16, 9))
    eps = np.linspace(0, len(loss), len(loss))
    plt.plot(eps, np.log10(loss), linewidth=1, label='loss')
    plt.plot(eps, np.log10(data_loss), linewidth=1, label='data_loss')
    plt.plot(eps, np.log10(residuals_loss), linewidth=1, label='residuals_loss')

    plt.title('train loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    # plt.yscale('log')
    plt.legend()

    # 设置刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # 设置图例字体大小
    plt.legend(loc=7, fontsize=16)

    plt.savefig(path_results+f'{country}_{type}_loss.pdf',dpi=600)
    plt.close()


def plot_result(country, data, data_type, realdata=None, I_pre=None):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(data),len(data)+1)[:-1]
    name = data_type
    y = 'Numbers of individuals'
    if data_type=='R0':
        data_type = '$R_{0}$'
        name = 'R0'
        y = name
    elif data_type == 'beta':
        data_type = '$beta$'
        name = 'beta'
        y = name
    elif data_type == 'gamma':
        data_type = '$gamma$'
        name = 'gamma'
        y = name

    plt.plot(t, data, color ='black' ,label=data_type)
    if not realdata is None:
        if data_type == 'C':
            plt.plot(t, realdata, color ='red' ,label='Confirmed')
            name = 'C_Confirmed'
        else:
            plt.plot(t, realdata, color ='red' ,label='Deaths')
            name = 'D_Deaths'

    if not I_pre is None:
      plt.plot(t, I_pre, color ='green' ,label='I')
      name = 'C_Confirmed_I'   

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel(y, fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{name}_result.pdf', dpi=600)
    plt.close()


def save_results(country,date,r0,beta,gamma,st,it,ct,rt,dt,confirmed,deaths):
    pf_pinn = pd.DataFrame({'date':date,'R0':r0,'beta':beta,'gamma':gamma,'susceptibles': st, 'infectives': it, 
    'confirmed': ct, 'recovered': rt, 'deaths': dt,'confirmed_real':confirmed,'deaths_real':deaths})
    pf_pinn.to_csv(path_results+f"{country}_pinn_result.csv", index=False, sep=',')


def save_results_fixed(country,date,st,it,rt,S,I,R,S_ode,I_ode,R_ode,train_size):
    pf_pinn = pd.DataFrame({'date':date,'susceptibles': st, 'infectives': it, 'removed': rt, 'S': S, 'I': I, 'R':R,'S_ode':S_ode,'I_ode':I_ode
    ,'R_ode':R_ode})
    pf_pinn.to_csv(path_results+f"{country}_{train_size}_pinn_fixed_result.csv", index=False, sep=',')


def save_results_fixed_parameters(country,date,st,it,rt,inewt,ict,Ic,Inew,S_ode,I_ode,R_ode,Ic_ode):
    pf_pinn = pd.DataFrame({'date':date,'susceptibles': st, 'infectives': it, 'removed': rt, 'infectives_cumulative': ict,'infectives_new': inewt, 'Ic_raw': Ic, 'Inew_raw': Inew,
    'S_ode': S_ode, 'I_ode': I_ode, 'R_ode': R_ode, 'Ic_ode': Ic_ode})
    pf_pinn.to_csv(path_results+f"{country}_pinn_result_fixed_parameters.csv", index=False, sep=',')


def save_parameters_learned(country,beta_raw,gamma_raw,beta,gamma,train_size):

    parameters_table = PrettyTable(['', 'init','learned'])
    parameters_table.add_row(['beta',beta_raw,beta])
    parameters_table.add_row(['gamma',gamma_raw,gamma])

    result_file_name = path_results+f'{country}_{train_size}_parameters_learned_result.txt'

    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(parameters_table)+'\n')

    fff.close()

def save_error_result_sir(country,s,i,r,st,it,rt,S_ode,I_ode,R_ode,train_size):
    #计算pinn mse值
    pinn_mse_s = metrics.mean_squared_error(s, st)
    pinn_mse_i = metrics.mean_squared_error(i, it)
    pinn_mse_r = metrics.mean_squared_error(r, rt)

    pinn_mse_sir = pinn_mse_s+pinn_mse_i+pinn_mse_r

    #计算pinn mae值
    pinn_mae_s = metrics.mean_absolute_error(s, st)
    pinn_mae_i = metrics.mean_absolute_error(i, it)
    pinn_mae_r = metrics.mean_absolute_error(r, rt)

    pinn_mae_sir = pinn_mae_s+pinn_mae_i+pinn_mae_r

    error_table = PrettyTable(['', 'mse','mae'])
    error_table.add_row(['sir',pinn_mse_sir,pinn_mae_sir])
    error_table.add_row(['s',pinn_mse_s,pinn_mae_s])
    error_table.add_row(['i',pinn_mse_i,pinn_mae_i])
    error_table.add_row(['r',pinn_mse_r,pinn_mae_r])

    result_file_name = path_results+f'{country}_{train_size}_error_result.txt'



    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(error_table)+'\n')

    fff.close()

    #计算pinn mse值
    ode_mse_s = metrics.mean_squared_error(s, S_ode)
    ode_mse_i = metrics.mean_squared_error(i, I_ode)
    ode_mse_r = metrics.mean_squared_error(r, R_ode)

    ode_mse_sir = ode_mse_s+ode_mse_i+ode_mse_r

    #计算pinn mae值
    ode_mae_s = metrics.mean_absolute_error(s, S_ode)
    ode_mae_i = metrics.mean_absolute_error(i, I_ode)
    ode_mae_r = metrics.mean_absolute_error(r, R_ode)

    ode_mae_sir = ode_mae_s+ode_mae_i+ode_mae_r

    ode_error_table = PrettyTable(['', 'mse','mae'])
    ode_error_table.add_row(['sir',ode_mse_sir,ode_mae_sir])
    ode_error_table.add_row(['s',ode_mse_s,ode_mae_s])
    ode_error_table.add_row(['i',ode_mse_i,ode_mae_i])
    ode_error_table.add_row(['r',ode_mse_r,ode_mae_r])

    ode_result_file_name = path_results+f'{country}_{train_size}_ode_error_result.txt'

    if os.path.exists(ode_result_file_name):
        os.remove(ode_result_file_name)

    fff = open(ode_result_file_name,'a')
    fff.write(str(ode_error_table)+'\n')

    fff.close()


def save_error_result(country,ct,dt,confirmed,deaths):
    #计算pinn mse值
    pinn_mse_C = metrics.mean_squared_error(ct[:], confirmed[:])
    pinn_mse_D = metrics.mean_squared_error(dt[:], deaths[:])

    pinn_mse_scird = pinn_mse_C+pinn_mse_D

    #计算pinn mae值
    pinn_mae_C = metrics.mean_absolute_error(ct[0:], confirmed[0:])
    pinn_mae_D = metrics.mean_absolute_error(dt[0:], deaths[0:])

    pinn_mae_sicrd = pinn_mae_C+pinn_mae_D

    error_table = PrettyTable(['', 'mae','mse'])
    error_table.add_row(['sicrd',pinn_mae_sicrd,pinn_mse_scird])
    error_table.add_row(['confirmed',pinn_mae_C,pinn_mse_C])
    error_table.add_row(['deaths',pinn_mae_D,pinn_mse_D])

    result_file_name = path_results+f'{country}_error_result.txt'

    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(error_table)+'\n')

    fff.close()


def save_results_time_dependent_parameters(country,date,st,it,rt,inewt,ict,Ic,Inew,S_ode,I_ode,R_ode,Ic_ode,beta,gamma):
    pf_pinn = pd.DataFrame({'date':date,'susceptibles': st, 'infectives': it, 'removed': rt, 'infectives_cumulative': ict,'infectives_new': inewt, 'Ic_raw': Ic, 'Inew_raw': Inew,
    'S_ode': S_ode, 'I_ode': I_ode, 'R_ode': R_ode, 'Ic_ode': Ic_ode, 'beta': beta, 'gamma': gamma})
    pf_pinn.to_csv(path_results+f"{country}_pinn_result_time_dependent_parameters.csv", index=False, sep=',')


def save_parameters_result(country,r0_raw,beta_raw,gamma_raw,r0,beta,gamma):

    parameters_table = PrettyTable(['', 'init','learned'])
    parameters_table.add_row(['R0',r0_raw,r0])
    parameters_table.add_row(['beta',beta_raw,beta])
    parameters_table.add_row(['gamma',gamma_raw,gamma])

    result_file_name = path_results+f'{country}_parameters_result.txt'

    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(parameters_table)+'\n')

    fff.close()


def save_fixed_parameters_result(country,beta_raw,gamma_raw,S0_raw,I0_raw,R0_raw,beta_learned,gamma_learned,S0_learned,I0_learned,R0_learned):

    parameters_table = PrettyTable(['', 'init','learned'])
    parameters_table.add_row(['beta',beta_raw,beta_learned])
    parameters_table.add_row(['gamma',gamma_raw,gamma_learned])
    parameters_table.add_row(['S0',S0_raw,S0_learned])
    parameters_table.add_row(['I0',I0_raw,I0_learned])
    parameters_table.add_row(['R0',R0_raw,R0_learned])

    result_file_name = path_results+f'{country}_fixed_parameters_result.txt'

    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(parameters_table)+'\n')

    fff.close()


def save_time_dependent_parameters_result(country,S0_raw,I0_raw,R0_raw,S0_learned,I0_learned,R0_learned):

    parameters_table = PrettyTable(['', 'init','learned'])
    parameters_table.add_row(['S0',S0_raw,S0_learned])
    parameters_table.add_row(['I0',I0_raw,I0_learned])
    parameters_table.add_row(['R0',R0_raw,R0_learned])

    result_file_name = path_results+f'{country}_time_dependent_parameters_result.txt'

    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    fff = open(result_file_name,'a')
    fff.write(str(parameters_table)+'\n')

    fff.close()


def plot_result_comparation(country, pre_data, data_type, real_data, ode_data,train_size):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(pre_data),len(pre_data)+1)[:-1]

    plt.plot(t, real_data, color ='black' ,label=f'{data_type}_Real') 
    plt.scatter(t[:train_size], real_data[:train_size], color ='black', marker='*', label=f'{data_type}_Train')  # type: ignore
    plt.plot(t, pre_data, color ='red' ,label=f'{data_type}_PINNs') 
    if data_type != 'Ic':
        plt.plot(t, ode_data, color ='green' ,label=f'{data_type}_SIR') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{data_type}_{train_size}_result.pdf', dpi=600)
    plt.close()


def plot_parameters_results(country, data_type, parameter):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(parameter),len(parameter)+1)[:-1]

    plt.plot(t, parameter, color ='red' ,label=f'{data_type}') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{data_type}_results.pdf', dpi=600)
    plt.close()


def plot_results(country, data_type, pre_data, ode_data, type):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(pre_data),len(pre_data)+1)[:-1]

    plt.plot(t, pre_data, color ='red' ,label=f'{data_type}_pinn') 
    plt.plot(t, ode_data, color ='green' ,label=f'{data_type}_sir') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{data_type}_{type}_results.pdf', dpi=600)
    plt.close()


def plot_show_results(country, data_type, pre_data, ode_data, type):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(pre_data),len(pre_data)+1)[:-1]

    plt.plot(t, pre_data, color ='red' ,label=f'{data_type}_pinn') 
    # plt.plot(t, ode_data, color ='green' ,label=f'{data_type}_sir') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{data_type}_{type}_results.pdf', dpi=600)
    plt.close()


def plot_results_comparation(country, data_type, real_data, pre_data, ode_data, train_size,type):
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16,9))
    t = np.linspace(0,len(pre_data),len(pre_data)+1)[:-1]

    plt.plot(t, real_data, color ='black' ,label=f'{data_type}_real') 
    plt.scatter(t[:train_size], real_data[:train_size], color ='black', marker='*', label=f'{data_type}_train')  # type: ignore
    plt.plot(t, pre_data, color ='red' ,label=f'{data_type}_pinn') 
    # plt.plot(t, ode_data, color ='green' ,label=f'{data_type}_sir') 

    plt.xlabel('Time t (days)', fontsize=25)
    plt.ylabel('Numbers of individuals', fontsize=25)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)

    plt.savefig(path_results+f'{country}_{data_type}_results_{type}_comparation.pdf', dpi=600)
    plt.close()


if __name__ == '__main__':
    print(timespans['Germany'])
    # path, paras = read_data_with_timespan('Germany',['2021-02-23','2021-07-01'])
    # print(paras)
    # print(path)
