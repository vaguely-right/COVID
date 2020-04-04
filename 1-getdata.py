#%% Get the Johns Hopkins data that includes Canadian provinces
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

can = pd.read_csv(infile)
can.drop(['Lat','Long'],axis=1,inplace=True)
can.rename(columns={'Country/Region' : 'Country', 'Province/State' : 'Province'},inplace=True)

can = can[can.Country=='Canada'].drop('Country',axis=1)
can.set_index('Province',inplace=True)
can = can.transpose()
can.drop(['Grand Princess','Diamond Princess','Recovered'],axis=1,inplace=True)
can.replace(0,np.nan,inplace=True)

#%% Get the world data
infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
wrld = pd.read_csv(infl)
wrld.set_index('date',inplace=True)
wrld.replace(0,np.nan,inplace=True)

#%% Get the Johns Hopkins US data
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
us = pd.read_csv(infile)
us.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1,inplace=True)
us.rename(columns={'Province_State':'State'},inplace=True)
us = us.groupby('State').sum()
us = us.transpose()
us.drop(['American Samoa','Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'],axis=1,inplace=True)
us.replace(0,np.nan,inplace=True)


