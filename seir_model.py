import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pwlf
from functools import reduce

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Get the world data
infile = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(infile)

#%% Use Italy as a test case
df = df[df.location=='Italy']

sns.lineplot(data=df,x='date',y='new_cases')
sns.lineplot(data=df,x='date',y='new_cases_smoothed')
sns.lineplot(data=df,x='date',y='new_tests')
sns.lineplot(data=df,x='date',y='new_tests_smoothed')
sns.lineplot(data=df,x='date',y='new_deaths')
sns.lineplot(data=df,x='date',y='new_deaths_smoothed')

df = df[['date','new_cases','new_deaths','new_tests','new_cases_smoothed','new_deaths_smoothed','new_tests_smoothed','total_cases','total_deaths']]

df.dropna(inplace=True)

#%% Get the Canadian data
casesfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/cases_timeseries_prov.csv'
mortfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/mortality_timeseries_prov.csv'
testsfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/testing_timeseries_prov.csv'

cases = pd.read_csv(casesfile)
deaths = pd.read_csv(mortfile)
tests = pd.read_csv(testsfile)

cases.rename(columns={'date_report':'date'},inplace=True)
deaths.rename(columns={'date_death_report':'date'},inplace=True)
tests.rename(columns={'date_testing':'date'},inplace=True)

indata = [cases,tests,deaths]

#df = cases.merge(tests,on=['province','date'])

df = reduce(lambda left,right: pd.merge(left,right,on=['province','date'],how='outer',suffixes=[None,'_x']),indata)
df.drop([i for i in df.columns.tolist() if '_x' in i],axis=1,inplace=True)

df = df[df.province=='Alberta']
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths']]
df.dropna(inplace=True)

#%% Fix the weekend zeros
df['a'] = (df.cases.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().cases,left_on='a',right_index=True,suffixes=['_x',''])

df['a'] = (df.testing.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().testing,left_on='a',right_index=True,suffixes=['_x',''])
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths']]

#%% Calculate the positivity
df['test_positivity'] = df.cases / df.testing

sns.lineplot(data=df,x='date',y='test_positivity')

#%% Determine the CFR values
sns.lineplot(data=df,x='cumulative_cases',y='cumulative_deaths')
df.plot.scatter('cumulative_cases','cumulative_deaths')

x = df.cumulative_cases.to_numpy()
y = df.cumulative_deaths.to_numpy()

cfr_fit = pwlf.PiecewiseLinFit(x,y)
res = cfr_fit.fit(3)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=100)
yHat = cfr_fit.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.show()

cfr = cfr_fit.slopes
df['cfr'] = np.sum([cfr[i]*((res[i]<=df.cumulative_cases) & (df.cumulative_cases<=res[i+1])) for i in range(3)],axis=0)

#%% 