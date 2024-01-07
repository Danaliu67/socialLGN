import pandas as pd
import scipy.io as sio
from scipy.io import savemat
from sklearn.model_selection import train_test_split
import world
import matplotlib.pyplot as plt

testSize = 0.2
n = world.n

d = sio.loadmat(f'./data/raw/ciao/rating_with_timestamp.mat')
prime = []
for val in d['rating']:
    user, item, rating, timestamp = val[0], val[1], val[3], val[5]
    prime.append([user, item, rating, timestamp])
df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])

min_timestamp = df['timestamp'].min()
max_timestamp = df['timestamp'].max()
interval = (max_timestamp - min_timestamp) / n
train_set, test_set = train_test_split(df, test_size=testSize, random_state=2020)

for idx in range(n):
    cutoff = min_timestamp + (idx+1) * interval
    filtered_df = train_set[train_set['timestamp'] < cutoff]
    filtered_data = filtered_df.to_numpy()
    mat_data = {'rating': filtered_data}
    savemat(f'./data/raw/ciao/rating_with_timestamp_{str(idx)}.mat', mat_data)

for idx in range(n):
    cutoff = min_timestamp + (idx+1) * interval
    filtered_df = test_set[test_set['timestamp'] < cutoff]
    filtered_data = filtered_df.to_numpy()
    mat_data = {'rating': filtered_data}
    savemat(f'./data/raw/ciao/rating_with_timestamp_test_{str(idx)}.mat', mat_data)
