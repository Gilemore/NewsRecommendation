# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = 'day_one_region_device.csv'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA, sep=',', usecols=[1,2,3,6,7,8,9,10])
data.columns = ['SessionId', 'ItemId','Time',"lat","long","Desktop","Mobile","Tablet"]
print(data.head())
# keep sessions with length >1 (avoid cold-start problem) 
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
# keep articles read more than 5 times
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
# keep sessions with length >=2 
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]
# now we have 156 unique articles and 34841 unique sessions, with each session at least 2 articles and each articles read at least 5 times. Length of data: 85710.

# we split the data set into train and test, noted that train and test have time order
split=data.Time.sort_values().quantile(.75)
session_max_times = data.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < split].index
session_test = session_max_times[session_max_times >= split].index
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]
# exclude those articles in test set but not in train set
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby('SessionId').size()
# still want to avoid the cold-start problem
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv('day_one_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv('day_one_test_full.txt', sep='\t', index=False)

# we split the training data into train and valid
split=train.Time.sort_values().quantile(.75)
session_max_times = train.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < split].index
session_valid = session_max_times[session_max_times >= split].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))

train_tr.to_csv('day_one_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv('day_one_train_valid.txt', sep='\t', index=False)
