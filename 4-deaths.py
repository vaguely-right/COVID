#%% Get the Johns Hopkins data that includes Canadian provinces
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

can = pd.read_csv(infile)
can.drop(['Lat','Long'],axis=1,inplace=True)
can.rename(columns={'Country/Region' : 'Country', 'Province/State' : 'Province'},inplace=True)

can = can[can.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()
can.drop(['Grand Princess','Diamond Princess','Recovered'],axis=1,inplace=True)
can.replace(0,np.nan,inplace=True)

#%% Get the world data
infl = r'https://covid.ourworldindata.org/data/ecdc/total_deaths.csv'
wrld = pd.read_csv(infl)
wrld.set_index('date',inplace=True)
wrld.replace(0,np.nan,inplace=True)

#%% Get the Johns Hopkins US data
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
us = pd.read_csv(infile)
us.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1,inplace=True)
us.rename(columns={'Province_State':'State'},inplace=True)
us = us.groupby('State').sum()
us = us.transpose()
us.drop(['American Samoa','Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'],axis=1,inplace=True)
us.replace(0,np.nan,inplace=True)
us.drop('Population',inplace=True)


#%% Find the countries that are at least five days ahead of Canada
ndays = wrld.Canada.gt(90).sum()+5
countries = list(wrld.max().index[wrld.gt(90).sum()>=ndays])
countries.append('Canada')
countries.remove('World')

wrld[countries].max().sort_values(ascending=False)

#%% Find the biggest provinces
can.gt(90).sum().sort_values(ascending=False)
provinces=['Alberta','Quebec','Ontario','British Columbia']
can[provinces].max().sort_values(ascending=False)

#%% Find the biggest states, the ones comparable to Canada
us.max().sort_values(ascending=False)
us.gt(90).sum().sort_values(ascending=False)

list1 = list(us.max().index[us.gt(wrld.Canada.max()).sum().gt(0)])
list2 = list(us.max().index[us.gt(20).sum().ge(wrld.Canada.gt(20).sum())])

#
states = list(set(list1+list2))
us[states].max().sort_values(ascending=False)

#%% Reframe the dataframes as days since 100th death
wrld = hundo(wrld,countries)
us = hundo(us,states)

#%% Calculate the five-day smoothed first derivative
wrldd1 = getd1(wrld,5)
usd1 = getd1(us,5)
pd.concat([wrldd1,usd1],axis=1).plot_bokeh(figsize=(1200,800),xlim=(4,30))


#%% Plot the world data as deaths vs growth rate
s1 = wrld.stack().to_frame().rename(columns={0:'Deaths'})
s2 = wrldd1.stack().to_frame().rename(columns={0:'Rate'})
stacked = pd.concat([s1,s2],axis=1).reset_index()
stacked.columns= ['Days','Country','Deaths','Rate']

sns.lineplot(data=stacked,x='Deaths',y='Rate',hue='Country')
plt.xscale('Log')

stacked.plot_bokeh.scatter(x='Deaths',y='Rate',category='Country',logx=True,figsize=(1200,800))




















