Nday_corr = []
for ndays in range(2,28):
    Nday_increase = wrld.diff(ndays).stack()
    Oneday_increase = wrld.diff(1).stack()
    Next14_days = wrld.diff(-14).stack()*-1
    df = pd.concat([Nday_increase,Oneday_increase,Next14_days],axis=1).dropna().reset_index(level=1).rename(columns={'level_1':'Country',0:'Nday_increase',1:'Oneday_increase',2:'Next14_days'})
    df = df[df.Nday_increase>0]
    df = df[df.Oneday_increase>0]
    df = df[df.Next14_days>0]
    df['Pct_increase'] = df.Oneday_increase / df.Nday_increase
    df['Log_next14'] = np.log(df.Next14_days)
    Nday_corr.append([ndays,np.corrcoef(df[['Pct_increase','Log_next14']].transpose().to_numpy())[0,1],len(df)])


Nday_corr
np.log(df)






df.plot.scatter('Nday_increase','Oneday_increase',logx=True,logy=True)


