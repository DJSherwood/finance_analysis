# import libraries
import numpy as np
import yfinance as yf
import pandas as pd
# import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
# pyo.init_notebook_mode()
import stumpy

# download data
name = 'ARCC'
ticker = yf.Ticker(name)
df = ticker.history(interval="1d",start='2024-01-01',end='2024-8-06')
df['Date'] = pd.to_datetime(df.index)
# calculate the midpoint
df['Midpoint'] = (df['High'] - df['Low']) / 2 + df['Low']
# calculate dollar total
df['DollarTotal'] = ( df['Midpoint'] * df['Volume'] ) / 1_000_000
# save data
df.to_csv('./stock_price.csv', sep=',', header=True, index=True)

window_size = 5
matrix_profile = stumpy.stump(df['High'], m=window_size)

## find a motif, it's nearest neighbor and location
motif_idx = np.argsort(matrix_profile[:, 0])[0]
nearest_neighbor_idx = matrix_profile[motif_idx, 1]
# find anomalies
discord_idx = np.argsort(matrix_profile[:, 0])[-1]
discord_neighbor = matrix_profile[discord_idx, 0]

print(f"The motif is located at index {motif_idx}")
print(f"The nearest neighbor is at {nearest_neighbor_idx}")
print("-------")
print(f"The discord is located at index {discord_idx}")
print(f"The nearest discord is at {discord_neighbor}")

## create a sequence of length of timeseries
tmain = np.linspace(0, len(df['High']), len(df['High']))

# plot
fig = go.FigureWidget(data=[
    go.Scatter(
        x=tmain,
        y=df['High'],
    )
])

fig.show()

## these functions are for finding the next value of a motif
## will have to convert to a percent change, though
from datetime import timedelta, date

def make_xy(mp, df, col, start, window=5):
    s = mp[start:start+window][0][1]
    y = df[col].iloc[s:s+window]
    t = np.arange(0, len(y))
    return [y, t]

def calc_error(df, col, ysub1, ymain):
    # calculate the last indexes for the two sequences
    final_neighbor_time = ysub1.index[(w-1)]
    final_main_time = ymain.index[(w-1)]
    # now, find the indexes for the above
    final_neighbor_index = df.index.get_indexer_for(df[df[col] == df[col].loc[str(final_neighbor_time)]].index)[0]
    final_main_index = df.index.get_indexer_for(df[df[col] == df[col].loc[str(final_main_time)]].index)[0]
    # find the values for the above indexes
    old1 = df[col].iloc[final_neighbor_index]
    new1 = df[col].iloc[final_main_index]
    old2 = df[col].iloc[final_neighbor_index+1]
    actual = df[col].iloc[final_main_index+1]
    # calculate error
    pred = ( new1 / old1 ) * old2
    err = ( np.abs( pred - actual ) / actual ) * 100
    return actual, pred, err

## here's the actual running of the thing
ysub1, ymain


i=142
w=5

ysub1, tsub1 = make_xy(matrix_profile, df, 'High', i)
ymain = df['High'].iloc[i:i+w]
tmain = np.arange(0, len(ymain))

actual, pred, err = calc_error(df, 'High', ysub1, ymain)

print(actual)
print(pred)
print(err)

## print
fig = go.FigureWidget(data=[
    go.Scatter(
        x=tmain,
        y=ymain.values,
    ),
    go.Scatter(
        x=tsub1,
        y=ysub1.values,
    ),
])

fig.show()

