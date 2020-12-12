#%% Look at individual cases a bit
df = get_ab_cases()

# Case fatality rate by age
df.pivot_table(index='age',columns='status',values='zone',aggfunc=len).apply(lambda x: x.Died/(x.Died+x.Recovered),axis=1)

# Infections by age by date
sns.lineplot(df.groupby(['date','age']).status.count().reset_index(),x='date',y='status',hue='age')

df.groupby(['date','age']).status.count().reset_index().plot.area()

df.pivot_table(index='date',columns='age',values='zone',aggfunc=len).plot.area()

