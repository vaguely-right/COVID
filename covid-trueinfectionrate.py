import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functools import reduce

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")


#%% Look at the relationship between positivity and infections found
rel = pd.DataFrame()
rel['positivity'] = [i/100 for i in range(1,100,1)]
rel['tests_per_case'] = [1 / i for i in rel.positivity]
rel['fraction_found'] = [np.exp(-1 * 8.80 * i) for i in rel.positivity]
rel['multiplier'] = [1 / i for i in rel.fraction_found]
rel['multiplier'] = [min(i,10) for i in rel.multiplier]
rel['fraction_p5'] = [np.exp(-1 * 9.5 * i) for i in rel.positivity]
rel['fraction_p95'] = [np.exp(-1 * 8.1 * i) for i in rel.positivity]

rel.plot.line(x='positivity',y=['fraction_found','fraction_p5','fraction_p95'],xticks=np.arange(0,1.05,0.05),yticks=np.arange(0,1.1,0.1))

rel.plot.line(x='positivity',y='multiplier',xlim=[0,0.30],ylim=[0,10])
rel.plot.line(x='positivity',y=['fraction_found','fraction_p5','fraction_p95'],xlim=[0,0.2],ylim=[0,1.0])


#%% Import the data sets
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

#sns.lineplot(data=df[df.province=='Alberta'].cases)
#sns.lineplot(data=df,x='date',y='cases',hue='province')

#%% Get the 7 day rolling averages
#provs = ['Alberta','BC','Ontario','Quebec']
#df = df[df.province.isin(provs)]
#df.set_index(['province','date'],inplace=True)


df['cases_7days'] = df.groupby('province').cases.rolling(7).mean().tolist()
df['tests_7days'] = df.groupby('province').testing.rolling(7).mean().tolist()
df['deaths_7days'] = df.groupby('province').deaths.rolling(7).mean().tolist()
df['active_7days'] = df.groupby('province').active_cases.rolling(7).mean().tolist()
df['recoveries_7days'] = df.groupby('province').recovered.rolling(7).mean().tolist()


df[df.province=='Alberta'][['cases','cases_7days']].plot()

df[df.province=='Alberta'][['testing','tests_7days']].plot()

df[df.province=='Alberta'][['deaths','deaths_7days']].plot()

df[df.province=='Alberta'][['active_cases','active_7days']].plot()

#sns.lineplot(data=df.pivot('date','province','cases_7days'))


#%% Calculate the adjusted cases based on testing
df = df[df.tests_7days>0]
df['positivity'] = df.cases_7days / df.tests_7days
df[df.province=='Alberta'].positivity.plot()

df['fraction_found'] = [np.exp(-1 * 8.80 * i) for i in df.positivity]
df['multiplier'] = [min(1/i,10) for i in df.fraction_found]
df[df.province=='Alberta'].multiplier.plot()

df['cases_est'] = df.cases_7days * df.multiplier

df[df.province=='Alberta'][['cases_est','cases_7days','positivity']].plot(secondary_y='positivity')
df[df.province=='BC'][['cases_est','cases_7days','positivity']].plot(secondary_y='positivity')
df[df.province=='Ontario'][['cases_est','cases_7days','positivity']].plot(secondary_y='positivity')
df[df.province=='Quebec'][['cases_est','cases_7days','positivity']].plot(secondary_y='positivity')
df[df.province=='Manitoba'][['cases_est','cases_7days','positivity']].plot(secondary_y='positivity')

#%% Look at some world data
infile = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(infile)

df.groupby('location').new_cases_smoothed.max().sort_values(ascending=False).head(20)

countries = df.groupby('location').new_cases_smoothed.max().sort_values(ascending=False).head(20).index.tolist()
countries.remove('World')
countries.append('Canada')

countries = ['Canada','United States','United Kingdom','Spain','Italy','France','Russia','South Africa','Colombia','Chile']

df.groupby('location').new_tests_smoothed.count().sort_values(ascending=False).head(20)

#df = df[df.location.isin(countries)]

#%% Estimate the true infections
df['positivity'] = df.new_cases_smoothed / df.new_tests_smoothed
df['fraction_found'] = [np.exp(-1 * 8.80 * i) for i in df.positivity]
df['multiplier'] = [min(1/i,10) for i in df.fraction_found]
df['cases_est'] = df.new_cases_smoothed * df.multiplier

c = 'Canada'

df[df.location==c][['date','cases_est','new_cases_smoothed','positivity']].plot(x='date',secondary_y='positivity',xlabel='Date',ylabel='Cases/day',title=c)

sns.lineplot(data=df[df.location==c],x='date',y='cases_est')

for c in countries:
    df[df.location==c][['date','cases_est','new_cases_smoothed','positivity']].plot(x='date',secondary_y='positivity',xlabel='Date',ylabel='Cases/day',title=c)


#%% Calculate Rt
rt = df.groupby('location').cases_est.rolling(7).mean().shift(-7) / df.groupby('location').cases_est.rolling(7).mean()

df['Rt'] = rt.rename('Rt').sort_index(level=1).tolist()


df[df.location=='Canada'].Rt.plot()

c = 'Canada'

df[df.location==c][['date','cases_est','new_cases_smoothed','Rt']].plot(x='date',secondary_y='Rt',xlabel='Date',ylabel='Cases/day',title=c)




#%% Look at the US data
infile = r'https://covidtracking.com/data/download/all-states-history.csv'



#%% Determine my own k value
#df_rates = df.loc[df.groupby('location')['total_tests'].idxmax().dropna()].groupby('location')[['total_cases','total_deaths','total_tests']].max().dropna()
infile = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(infile)

cols = ['total_cases','total_deaths','total_tests']
df_rates = df.loc[df[df.date>='2020-09-22'].groupby('location')['total_tests'].idxmax().dropna()].groupby('location')[cols].max().dropna()
df_rates = df_rates[df_rates.total_deaths>=1]

#df_rates.drop('World',inplace=True)

df_rates['cfr'] = df_rates.total_deaths / df_rates.total_cases
df_rates['positivity'] = df_rates.total_cases / df_rates.total_tests
df_rates['c'] = 1 / df_rates.positivity

df_rates = df_rates[df_rates.positivity < 0.2]

df_rates.plot.scatter(x='total_cases',y='total_deaths',logx=True,logy=True)

df_rates.plot.scatter(x='positivity',y='cfr',logy=True)
df_rates.plot.scatter(x='c',y='cfr',logy=True)
df_rates.plot.scatter(x='total_cases',y='total_deaths',logx=True,logy=True,c='positivity',colormap='rainbow')

X = df_rates.positivity.to_numpy().reshape(-1,1)
Y = np.log(df_rates.cfr.to_numpy()).reshape(-1,1)
W = np.log(df_rates.total_tests).to_numpy()
reg = LinearRegression().fit(X,Y,sample_weight=W)

reg.coef_
reg.intercept_
np.exp(reg.intercept_)

df_rates['log_cfr'] = np.log(df_rates.cfr)
df_rates.corr()[['cfr','log_cfr']]

sns.pairplot(df_rates)

countries = df_rates.index.values.tolist()

#%%
df_rates = df[df.location=='Canada'][['date']+cols].dropna()


#%% Try something else w/Permanence of Ratios
infile = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(infile)

cols = ['total_cases_per_million','total_deaths_per_million','total_tests_per_thousand']

df_rates = df.loc[df[df.date>='2020-09-24'].groupby('location')['total_tests'].idxmax().dropna()].groupby('location')[cols].max().dropna()
df_rates['positivity'] = df_rates.total_cases_per_million / df_rates.total_tests_per_thousand / 1000

df_rates = df_rates[df_rates.positivity < 0.2]

df_rates['case_fatality_rate'] = df_rates.total_deaths_per_million / df_rates.total_cases_per_million

df_rates['deaths_per_test'] = df_rates.total_deaths_per_million / df_rates.total_tests_per_thousand / 1000


c = df_rates.case_fatality_rate / (1 - df_rates.case_fatality_rate)
d = df_rates.total_deaths_per_million / (1000000 - df_rates.total_deaths_per_million)
t = df_rates.deaths_per_test / (1 - df_rates.deaths_per_test)

i = c * d / t

df_rates['infection_fatality_rate'] = i / (1 + i)


df_rates['const'] = df_rates.total_cases_per_million * df_rates.positivity / df_rates.total_tests_per_thousand / 1000


