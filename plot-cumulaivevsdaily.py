pd.concat([
        wrld[countries].diff(7).div(7).stack().rename('7 day average'),
        wrld[countries].stack().rename('Cumulative')
        ],axis=1).reset_index(level=1).plot_bokeh.scatter(x='Cumulative',y='7 day average',category='level_1',figsize=(1200,800))

pd.concat([
        us[states].diff(7).div(7).stack().rename('7 day average'),
        us[states].stack().rename('Cumulative')
        ],axis=1).reset_index(level=1).plot_bokeh.scatter(x='Cumulative',y='7 day average',category='State',figsize=(1200,800))

pd.concat([
        can.diff(7).div(7).stack().rename('7 day average'),
        can.stack().rename('Cumulative')
        ],axis=1).reset_index(level=1).plot_bokeh.scatter(x='Cumulative',y='7 day average',category='Province',figsize=(1200,800))
