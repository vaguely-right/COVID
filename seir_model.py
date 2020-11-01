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
actfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/active_timeseries_prov.csv'
casesfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/cases_timeseries_prov.csv'
mortfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/mortality_timeseries_prov.csv'
recfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/recovered_timeseries_prov.csv'
testsfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/testing_timeseries_prov.csv'

active = pd.read_csv(actfile)
cases = pd.read_csv(casesfile)
deaths = pd.read_csv(mortfile)
recoveries = pd.read_csv(recfile)
tests = pd.read_csv(testsfile)

active.rename(columns={'date_active':'date'},inplace=True)
cases.rename(columns={'date_report':'date'},inplace=True)
deaths.rename(columns={'date_death_report':'date'},inplace=True)
recoveries.rename(columns={'date_recovered':'date'},inplace=True)
tests.rename(columns={'date_testing':'date'},inplace=True)

indata = [cases,tests,deaths,active,recoveries]

#df = cases.merge(tests,on=['province','date'])

df = reduce(lambda left,right: pd.merge(left,right,on=['province','date'],how='outer',suffixes=[None,'_x']),indata)
df.drop([i for i in df.columns.tolist() if '_x' in i],axis=1,inplace=True)

df = df[df.province=='Alberta']
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths','active_cases','recovered']]
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#%% Fix the weekend zeros
df['a'] = (df.cases.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().cases,left_on='a',right_index=True,suffixes=['_x',''])

df['a'] = (df.testing.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().testing,left_on='a',right_index=True,suffixes=['_x',''])
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths','active_cases','recovered']]

#%% Calculate the positivity
df['test_positivity'] = df.cases / df.testing

#sns.lineplot(data=df,x='date',y='test_positivity')

#%% Determine the CFR values
#sns.lineplot(data=df,x='cumulative_cases',y='cumulative_deaths')
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


#%% Get a rough Rt value using seven-day rolling averages
df['rt'] = df.cases.rolling(14).mean().shift(-14)/df.cases.rolling(14).mean()
df.rt.fillna(method='bfill',inplace=True)
df.rt.fillna(method='ffill',inplace=True)

#%% Function for going through the SEIR model

def eukrd(beta,p,cfr,alpha,gamma,k,e_0,i_u_0,i_k_0,r_0,d_0):
# Initialize all of the arrays
    e = np.empty(shape=beta.shape)
    e[:] = np.nan
    d_e = np.empty(shape=beta.shape)
    d_e[:] = np.nan
    i_u = np.empty(shape=beta.shape)
    i_u[:] = np.nan
    d_i_u = np.empty(shape=beta.shape)
    d_i_u[:] = np.nan
    i_k = np.empty(shape=beta.shape)
    i_k[:] = np.nan
    d_i_k = np.empty(shape=beta.shape)
    d_i_k[:] = np.nan
    r = np.empty(shape=beta.shape)
    r[:] = np.nan
    d_r = np.empty(shape=beta.shape)
    d_r[:] = np.nan
    d = np.empty(shape=beta.shape)
    d[:] = np.nan
    d_d = np.empty(shape=beta.shape)
    d_d[:] = np.nan
    d_c = np.empty(shape=beta.shape)
    d_c[:] = np.nan

    # Initialize time step 0
    e[0] = e_0
    i_u[0] = i_u_0
    i_k[0] = i_k_0
    r[0] = r_0
    d[0] = d_0
    d_e[0] = 0
    d_i_u[0] = 0
    d_i_k[0] = 0
    d_c[0] = 0
    d_r[0] = 0
    d_d[0] = 0

# Work through the time steps
    for t in range(1,len(beta)):
        d_e[t] = beta[t] * ( i_u[t-1] + i_k[t-1] ) - alpha * e[t-1]
        d_i_u[t] = alpha * e[t-1] - gamma * i_u[t-1] - np.exp(-k * p[t]) * i_u[t-1]
        d_i_k[t] = np.exp(-k * p[t]) * i_u[t-1] - gamma * i_k[t-1]
        d_c[t] = np.exp(-k * p[t]) * i_u[t-1]
        d_r[t] = gamma * (i_u[t-1] + i_k[t-1] * (1-cfr[t]))
        d_d[t] = gamma * i_k[t-1] * cfr[t]
        e[t] = e[t-1] + d_e[t]
        i_u[t] = i_u[t-1] + d_i_u[t]
        i_k[t] = i_k[t-1] + d_i_k[t]
        r[t] = r[t-1] + d_r[t]
        d[t] = d[t-1] + d_d[t]

    return e,i_u,i_k,r,d,d_e,d_i_u,d_i_k,d_c,d_r,d_d



#%% Set some parameters and try running it
# The fixed variables
alpha = 1/2
gamma = 1/12
k = -8.8

# The initial state
e_0 = 50
i_u_0 = 20
i_k_0 = df.active_cases.iloc[0]
r_0 = df.recovered.iloc[0]
d_0 = df.cumulative_deaths.iloc[0]

# The variable parameters
beta = df.rt.to_numpy() * gamma
p = df.test_positivity.to_numpy()
p[-1] = p[-2]
cfr = df.cfr.to_numpy()

# The true values to calibrate to
d_c = df.cases.to_numpy()
d_d_true = df.deaths.to_numpy()

e,i_u,i_k,r,d,d_e,d_i_u,d_i_k,d_c,d_r,d_d = eukrd(beta,p,cfr,alpha,gamma,k,e_0,i_u_0,i_k_0,r_0,d_0)

df_eukrd = pd.DataFrame({'e' : e,
              'i_u' : i_u,
              'i_k' : i_k,
              'r' : r,
              'd' : d,
              'd_e' : d_e,
              'd_i_u' : d_i_u,
              'd_i_k' : d_i_k,
              'd_c' : d_c,
              'd_r' : d_r,
              'd_d' : d_d})

pd.concat([df,df_eukrd],axis=1)[['cases','d_c']].plot()




















