import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import math
import numpy as np
from datetime import datetime
import json
import bisect

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)
exception_name_columns = config['exception_name']['train']

n_exceptions_train = pd.read_csv(config['data']['features']['aug_train']['n_exceptions_recent_week_filter'])
n_exceptions_test = pd.read_csv(config['data']['features']['B_test']['n_exceptions_recent_week'])

n_exceptions_train = n_exceptions_train[sum([n_exceptions_train[exception_name+'_cnt_'+str(i)] for exception_name in exception_name_columns for i in range(0, 8)]) > 0]

for exception_name in exception_name_columns:

    n_exceptions_train[exception_name+'_rate_0']\
            = (n_exceptions_train[exception_name+'_cnt_0'] * 23 / n_exceptions_train[exception_name+'_cnt_1'].clip(lower=0.01) - 1.5).clip(lower=0, upper=10)
    n_exceptions_test[exception_name+'_rate_0']\
            = (n_exceptions_test[exception_name+'_cnt_0'] * 23 / n_exceptions_test[exception_name+'_cnt_1'].clip(lower=0.01) - 1.5).clip(lower=0, upper=10)
    n_exceptions_train[exception_name+'_cnt_1'] = n_exceptions_train[exception_name+'_cnt_1'] + n_exceptions_train[exception_name+'_cnt_0']
    n_exceptions_test[exception_name+'_cnt_1'] = n_exceptions_test[exception_name+'_cnt_1'] + n_exceptions_test[exception_name+'_cnt_0']
    
    for i in range(1, 7):
        n_exceptions_train[exception_name+'_rate_'+str(i)]\
            = (n_exceptions_train[exception_name+'_cnt_'+str(i)] / n_exceptions_train[exception_name+'_cnt_'+str(i + 1)].clip(lower=0.01) - 1.5).clip(lower=0, upper=10)
        n_exceptions_test[exception_name+'_rate_'+str(i)]\
            = (n_exceptions_test[exception_name+'_cnt_'+str(i)] / n_exceptions_test[exception_name+'_cnt_'+str(i + 1)].clip(lower=0.01) - 1.5).clip(lower=0, upper=10)

    n_exceptions_train[exception_name+'_sum_rate'] = sum(n_exceptions_train[exception_name+'_rate_'+str(i)] for i in range(0, 7))
    n_exceptions_test[exception_name+'_sum_rate'] = sum(n_exceptions_test[exception_name+'_rate_'+str(i)] for i in range(0, 7))

sum_rate_exceptions_train = n_exceptions_train[['nc_ip', 'sample_time', 'nc_down_label'] + [exception_name + '_sum_rate' for exception_name in exception_name_columns]]
sum_rate_exceptions_test = n_exceptions_test[['nc_ip', 'sample_time', 'nc_down_label'] + [exception_name + '_sum_rate' for exception_name in exception_name_columns]]

sum_rate_exceptions_train.to_csv(config['data']['features']['aug_train']['sum_rate_exceptions_recent_week_filter'], index=False)
# sum_rate_exceptions_test.to_csv(config['data']['features']['B_test']['sum_rate_exceptions_recent_week_filter'], index=False)

