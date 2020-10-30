import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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

df = df[['date','new_cases','new_deaths','new_tests','new_cases_smoothed','new_deaths_smoothed','new_tests_smoothed']]

#%% Calculate the positivity
df['test_positivity'] = df.new_cases / df.new_tests

sns.lineplot(data=df,x='date',y='test_positivity')
