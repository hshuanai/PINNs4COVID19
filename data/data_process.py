# -*- coding: utf-8 -*-

import os

import constants
import pandas as pd

if not os.path.exists(constants.path_plots_dataprocessed):
    os.makedirs(constants.path_plots_dataprocessed)

if not os.path.exists(constants.path_results):
    os.makedirs(constants.path_results)

if not os.path.exists(constants.path_dataprocessed):
    os.makedirs(constants.path_dataprocessed)

# 加载数据
pf_confirmed = pd.read_csv(constants.path_confirmed)
pf_deaths = pd.read_csv(constants.path_deaths)
pf_recovered = pd.read_csv(constants.path_recovered)

def choose_data_4_SIR(countries,timespans,populations):

    # 筛选数据（国家，时间，Ic,Inew,Recovered,Death）Inew= Ic(t)-Ic(t-1). I = Ic-Rc-Dc. Dc=D. R = Rc = Rc+Dc
    for i, country in enumerate(countries):
        # 筛选指定列数据
        confirmed = pf_confirmed.loc[pf_confirmed['Country/Region'] == country].iloc[:, 4:]
        deaths = pf_deaths.loc[pf_deaths['Country/Region'] == country].iloc[:, 4:]
        recovered = pf_recovered.loc[pf_deaths['Country/Region'] == country].iloc[:, 4:]

        rawdata = confirmed.append(recovered).append(deaths).transpose()
        
        rawdata['Ic'] = rawdata.iloc[:, 0]
        rawdata['Inew'] = rawdata.iloc[:, 0].diff(1)
        rawdata['R'] = rawdata.iloc[:, 1]
        rawdata['D'] = rawdata.iloc[:, 2]

        data = rawdata[['Ic', 'Inew', 'R', 'D']].dropna(axis=0, how='any')

        data['R_4_SIR'] = data['R'] + data['D']
        data['I_4_SIR'] = data['Ic'] - data['R_4_SIR']
        data['S_4_SIR'] = populations[i] - (data['R_4_SIR'] + data['I_4_SIR'])

        # 时间格式处理
        data = data.reset_index().rename(columns={'index': 'date'})
        data['date'] = pd.to_datetime(data['date'])  # date转为时间格式

        # 选取时间段数据
        data.set_index(["date"], inplace=True)
        # 时间可以设置为为每个国家分配不同的时间。
        data = data[timespans[country][0]:timespans[country][1]]

        constants.plots(country, data['Inew'], 'Inew', data_type='unrolled')
        constants.plots(country, data['D'], 'D', data_type='unrolled')
        constants.plots(country, data['Ic'], 'Ic', data_type='unrolled')
        constants.plots(country, data['R'], 'R', data_type='unrolled')

        constants.plots(country, data['S_4_SIR'], 'Susceptible', data_type='unrolled')
        constants.plots(country, data['I_4_SIR'], 'Infectious', data_type='unrolled')
        constants.plots(country, data['R_4_SIR'], 'Removed', data_type='unrolled')

        # 滑动窗口处理
        data_rolled = data.rolling(window=7).mean()[6:]
        data_rolled['date'] = data.index[6:]
        data_rolled.set_index(["date"], inplace=True)

        constants.plots(country, data_rolled['Inew'], 'Inew', data_type='rolled')
        constants.plots(country, data_rolled['D'], 'D', data_type='rolled')
        constants.plots(country, data_rolled['Ic'], 'Ic', data_type='rolled')
        constants.plots(country, data_rolled['R'], 'R', data_type='rolled')

        # constants.plots(country, data_rolled['S_4_SIR'], 'Susceptible', data_type='rolled')
        # constants.plots(country, data_rolled['I_4_SIR'], 'Infectious', data_type='rolled')
        # constants.plots(country, data_rolled['R_4_SIR'], 'Removed', data_type='rolled')

        # 保存数据 国家_人口_时间范围_数据长度.csv
        file = constants.path_dataprocessed + \
            f"{country}_{populations[i]}_{timespans[country][0]}_{timespans[country][1]}_{len(data_rolled)}_realdata.csv"
        data_rolled.to_csv(file, index=True, sep=',')



if __name__ == '__main__':
    choose_data_4_SIR(constants.countries,constants.timespans,constants.populations)