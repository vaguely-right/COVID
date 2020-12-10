import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas_bokeh
pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Import the data
infile = r'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
df = pd.read_csv(infile)

df.columns.tolist()
df.iso_code.value_counts()

df = df[df.total_tests.notna()]

# Look at the testing methods
df.groupby(['tests_units']).iso_code.nunique()

# Calculate the case fatality rate
df['case_fatality_rate'] = df.total_deaths / df.total_cases * 100
df['case_fatality_ratio'] = df.total_deaths / df.total_cases
df['test_ratio'] = df.total_tests_per_thousand.div(1000)
df['positivity_ratio'] = df.total_tests / df.total_cases
df['log_case_fatality_ratio'] = np.log10(df.case_fatality_ratio)
df['log_positivity_ratio'] = np.log10(df.positivity_ratio)
df['log_test_ratio'] = np.log10(df.test_ratio)

#%% Look at positivity, 
# Start by ignoring method and doing it naive
# Find, for each country, the date that it reached 100, 1000, and 10000 deaths

df[df.total_deaths>=100].iso_code.unique()
df[df.total_deaths>=1000].iso_code.unique()
df[df.total_deaths>=10000].iso_code.unique()

df.groupby('iso_code').total_deaths.max().sort_values(ascending=False)

dat = []
for value in 10**np.arange(1,5):
    dat.append(df.loc[df.query('total_deaths>='+str(value)).groupby('iso_code').total_deaths.idxmin()])

df1 = pd.concat(dat,axis=0)

sns.scatterplot(x='total_tests_per_thousand',y='case_fatality_rate',data=df1)

sns.scatterplot(x='log_test_ratio',y='log_case_fatality_ratio',data=df1)

sns.lmplot(x='log_test_ratio',y='log_case_fatality_ratio',data=df1)
sns.lmplot(x='log_test_ratio',y='log_case_fatality_ratio',data=df1,hue='tests_units')

#%% Try just a single data point per country
df1 = df.loc[df.groupby('iso_code').total_deaths.idxmax()]
























