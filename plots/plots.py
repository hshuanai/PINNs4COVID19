# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # Python画图工具 
from matplotlib.gridspec import GridSpec     # 利用网格确定图形的位置

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_it(t,train_size,name,pinn_result,sir_result,data,data_type,country):
  plt.figure(figsize=(16,9))

  # 真实值
  plt.plot(t, data, color ='black', linewidth = 2, label=f"{name}"+'$_{Data}$')
  # 训练集
  plt.scatter(t[:train_size], data[:train_size], color ='black', linewidth = 2, marker='*', label=f"{name}"+'$_{Train}$')  # type: ignore
  # sir
  plt.plot(t, sir_result, linestyle = '--', color = 'g', linewidth = 2, label=f"{name}"+'$_{SIR}$')
  # pinn
  plt.plot(t, pinn_result, color ='r', linewidth = 2, label=f"{name}"+'$_{PINN}$')

  plt.xlabel('Time t (days)', fontsize=30)
  plt.ylabel('Numbers of individuals', fontsize=30)

  plt.xticks(fontsize=30)
  plt.yticks(fontsize=30)

  plt.legend(fontsize=25)
  plt.savefig('plots2paper/'+f'{country}_{train_size}_for_{name}_{data_type}.pdf',dpi=600)
  plt.close()


def plot_all(t,train_size,name,pinn_result,sir_result,data,data_type,country):
    fig = plt.figure(figsize=(40,20))

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.xlabel('Time t (days)', fontsize=50)
    plt.ylabel('Numbers of individuals', fontsize=50)

    plt.xticks([])
    plt.yticks([])

    gs = GridSpec(40, 40)

    # 第一个子图
    ax1 = fig.add_subplot(gs[8:32, 2:20]) 

    ax1.set_title('I',fontsize=50)
    # 真实值
    ax1.plot(t, data[1], color ='black', linewidth = 2, label=f"{name[1]}"+'$_{Data}$')
    # 训练集
    ax1.scatter(t[:train_size], data[1][:train_size], color ='black', linewidth = 2, marker='*', label=f"{name[1]}"+'$_{Train}$')  # type: ignore
    # sir
    ax1.plot(t, sir_result[1], linestyle = '--', color = 'g', linewidth = 2, label=f"{name[1]}"+'$_{SIR}$')
    # pinn
    ax1.plot(t, pinn_result[1], color ='r', linewidth = 2, label=f"{name[1]}"+'$_{PINN}$')

#     ax1.set_xlabel('I',fontsize=50)
    ax1.tick_params(labelsize=50)
    ax1.legend(fontsize=30)

    # 第二个子图
    ax2 = fig.add_subplot(gs[1:17, 23:39])   

    ax2.set_title('S',fontsize=50)
    # 真实值
    ax2.plot(t, data[0], color ='black', linewidth = 2, label=f"{name[0]}"+'$_{Data}$')
    # 训练集
    ax2.scatter(t[:train_size], data[0][:train_size], color ='black', linewidth = 2, marker='*', label=f"{name[0]}"+'$_{Train}$')  # type: ignore
    # sir
    ax2.plot(t, sir_result[0], linestyle = '--', color = 'g', linewidth = 2, label=f"{name[0]}"+'$_{SIR}$')
    # pinn
    ax2.plot(t, pinn_result[0], color ='r', linewidth = 2, label=f"{name[0]}"+'$_{PINN}$')

#     ax2.set_xlabel('S',fontsize=50)
    ax2.tick_params(labelsize=50)
    ax2.legend(fontsize=30)

    # 第三个子图
    ax3 = fig.add_subplot(gs[22:38, 23:39]) 
    
    ax3.set_title('R',fontsize=50)
    # 真实值
    ax3.plot(t, data[2], color ='black', linewidth = 2, label=f"{name[2]}"+'$_{Data}$')
    # 训练集
    ax3.scatter(t[:train_size], data[2][:train_size], color ='black', linewidth = 2, marker='*', label=f"{name[2]}"+'$_{Train}$')  # type: ignore
    # sir
    ax3.plot(t, sir_result[2], linestyle = '--', color = 'g', linewidth = 2, label=f"{name[2]}"+'$_{SIR}$')
    # pinn
    ax3.plot(t, pinn_result[2], color ='r', linewidth = 2, label=f"{name[2]}"+'$_{PINN}$')

#     ax3.set_xlabel('R',fontsize=50)
    ax3.tick_params(labelsize=50)
    ax3.legend(fontsize=30)

    plt.savefig('plots2paper/'+f'{country}_{train_size}_for_{train_size}_{data_type}.pdf',dpi=600)
    plt.close(fig)



if __name__ == '__main__':
    print('----------------------------------------------------------------------------------------------------')

    # load result data
    country = 'Germany'
    pf_50 = pd.read_csv('/home/shuai/Remote/remote_coding/pinn4sir/results(paper)/Germany_50_pinn_fixed_result.csv')
    pf_30 = pd.read_csv('/home/shuai/Remote/remote_coding/pinn4sir/results(paper)/Germany_30_pinn_fixed_result.csv')
    pf_40 = pd.read_csv('/home/shuai/Remote/remote_coding/pinn4sir/results(paper)/Germany_40_pinn_fixed_result.csv')
    
    t = np.linspace(0,len(pf_50),len(pf_50)+1)[:-1]
    # plot_it(t,50,"S",pf_50['susceptibles'],pf_50['S_ode'],pf_50['S'],'realdata',country)    
    # plot_it(t,50,"I",pf_50['infectives'],pf_50['I_ode'],pf_50['I'],'realdata',country)  
    # plot_it(t,50,"R",pf_50['removed'],pf_50['R_ode'],pf_50['R'],'realdata',country)
    
    # plot_it(t,40,"S",pf_40['susceptibles'],pf_40['S_ode'],pf_40['S'],'realdata',country)    
    # plot_it(t,40,"I",pf_40['infectives'],pf_40['I_ode'],pf_40['I'],'realdata',country)  
    # plot_it(t,40,"R",pf_40['removed'],pf_40['R_ode'],pf_40['R'],'realdata',country)

    # plot_it(t,30,"S",pf_30['susceptibles'],pf_30['S_ode'],pf_30['S'],'realdata',country)    
    # plot_it(t,30,"I",pf_30['infectives'],pf_30['I_ode'],pf_30['I'],'realdata',country)  
    # plot_it(t,30,"R",pf_30['removed'],pf_30['R_ode'],pf_30['R'],'realdata',country)  

    print('----------------------------------------------------------------------------------------------------')
    
    plot_all(t,50,['S','I','R'],[pf_50['susceptibles'],pf_50['infectives'],pf_50['removed']],
            [pf_50['S_ode'],pf_50['I_ode'],pf_50['R_ode']],[pf_50['S'],pf_50['I'],pf_50['R']],'realdata',country)  
    
    plot_all(t,40,['S','I','R'],[pf_40['susceptibles'],pf_40['infectives'],pf_40['removed']],
            [pf_40['S_ode'],pf_40['I_ode'],pf_40['R_ode']],[pf_40['S'],pf_40['I'],pf_40['R']],'realdata',country) 

    plot_all(t,30,['S','I','R'],[pf_30['susceptibles'],pf_30['infectives'],pf_30['removed']],
            [pf_30['S_ode'],pf_30['I_ode'],pf_30['R_ode']],[pf_30['S'],pf_30['I'],pf_30['R']],'realdata',country) 
