import pandas as pd

infl = r'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
wrld = pd.read_csv(infl)
wrld.set_index('date',inplace=True)
wrld.replace(0,np.nan,inplace=True)

ndays = wrld.Canada.gt(90).sum()+4
list1 = list(wrld.max().index[wrld.gt(90).sum()>=ndays])
list2 = list(wrld.max().index[wrld.ge(wrld.Canada.max()).sum().gt(0)])
countries = list(set(list1+list2))
#countries.append('Canada')
countries.remove('World')
countries.remove('China')

wrld = hundo(wrld,countries,90)

#%%
df = pd.DataFrame({'Nday_pct_increase':wrld.pct_change(14,fill_method=None).stack(),'Nextday_pct_increase':wrld.pct_change(1,fill_method=None).stack()})


#%% Determine the optimal number of days
Nday_corr = []
for ndays in range(2,28):
    Nday_increase = wrld.diff(ndays).stack()
    Oneday_increase = wrld.diff(1).stack()
    Next14_days = wrld.diff(-14).stack()*-1
    Next14_days_pct = wrld.pct_change(-14,fill_method=None).stack()*-1
    Next_oneday = wrld.diff(-1).stack()*-1
    df = pd.concat([Nday_increase,Oneday_increase,Next14_days],axis=1).dropna().reset_index(level=1).rename(columns={'level_1':'Country',0:'Nday_increase',1:'Oneday_increase',2:'Next14_days'})
    df = df[df.Nday_increase>0]
    df = df[df.Oneday_increase>0]
    df = df[df.Next14_days>0]
    df['Pct_increase'] = df.Oneday_increase / df.Nday_increase
    df['Log_next14'] = np.log(df.Next14_days)
    Nday_corr.append([ndays,np.corrcoef(df[['Pct_increase','Log_next14']].transpose().to_numpy())[0,1],len(df)])


df = pd.DataFrame(Nday_corr,columns=['Ndays','Correlation','Nsamples'])
df.Correlation.plot()


#%% Use a specified number of days
ndays = 14
Nday_increase = wrld.diff(ndays).stack()
Oneday_increase = wrld.diff(1).stack()
Next14_days = wrld.diff(-14).stack()*-1
df = pd.concat([Nday_increase,Oneday_increase,Next14_days],axis=1).dropna().reset_index(level=1).rename(columns={'level_1':'Country',0:'Nday_increase',1:'Oneday_increase',2:'Next14_days'})
df = df[df.Nday_increase>0]
df = df[df.Oneday_increase>0]
df = df[df.Next14_days>0]
df['Pct_increase'] = df.Oneday_increase / df.Nday_increase
df['Log_next14'] = np.log(df.Next14_days)



df.plot.scatter('Nday_increase','Oneday_increase',logx=True,logy=True)


#%% Plot the average 3-day increase as a ratio to the previous 14-day average increase
df = pd.DataFrame({'Increase17Days':wrld.diff(17).stack(),'Increase3Days':wrld.diff(3).stack()})
df['Ratio'] = df.Increase3Days / (df.Increase17Days-df.Increase3Days) * 14/3

df.Ratio.unstack().plot_bokeh(figsize=(1200,800))

#%% Do this for Canada
df = pd.DataFrame({'Increase17Days':can.diff(17).stack(),'Increase3Days':can.diff(3).stack()})
df['Ratio'] = df.Increase3Days / (df.Increase17Days-df.Increase3Days) * 14/3

df.Ratio.unstack().plot_bokeh(figsize=(1200,800))


#%% And the US
df = pd.DataFrame({'Increase17Days':us.diff(17).stack(),'Increase3Days':us.diff(3).stack()})
df['Ratio'] = df.Increase3Days / (df.Increase17Days-df.Increase3Days) * 14/3

df.Ratio.unstack().plot_bokeh(figsize=(1200,800))















