import pandas as pd
import scipy.io as sio
from scipy.io import savemat
from sklearn.model_selection import train_test_split

testSize = 0.2
timestamp_cutoff = [1041379200, 1167609600, 1302591600]

d = sio.loadmat(f'./data/raw/ciao/rating_with_timestamp.mat')
prime = []
for val in d['rating']:
    user, item, rating, timestamp = val[0], val[1], val[3], val[5]
    prime.append([user, item, rating, timestamp])
df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
print(df)

train_set, test_set = train_test_split(df, test_size=testSize, random_state=2020)

for idx, cutoff in enumerate(timestamp_cutoff):
    filtered_df = train_set[train_set['timestamp'] < cutoff]
    filtered_data = filtered_df.to_numpy()
    mat_data = {'rating': filtered_data}
    savemat(f'./data/raw/ciao/rating_with_timestamp_{str(idx)}.mat', mat_data)


filtered_data = test_set.to_numpy()
mat_data = {'rating': filtered_data}
savemat('./data/raw/ciao/rating_with_timestamp_test.mat', mat_data)
