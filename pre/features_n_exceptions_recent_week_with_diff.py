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

n_exceptions_train = pd.read_csv(config['data']['features']['train']['n_exceptions_recent_week'])
n_exceptions_test = pd.read_csv(config['data']['features']['test']['n_exceptions_recent_week'])

for exception_name in exception_name_columns:
    n_exceptions_train[exception_name+'_inc_1'] = n_exceptions_train[exception_name+'_cnt_0'] + n_exceptions_train[exception_name+'_cnt_1'] - n_exceptions_train[exception_name+'_cnt_2']
    n_exceptions_test[exception_name+'_inc_1'] = n_exceptions_test[exception_name+'_cnt_0'] + n_exceptions_test[exception_name+'_cnt_1'] - n_exceptions_test[exception_name+'_cnt_2']

    for i in range(2, 7):
        n_exceptions_train[exception_name+'_inc_'+str(i)] = n_exceptions_train[exception_name+'_cnt_'+str(i)] - n_exceptions_train[exception_name+'_cnt_'+str(i + 1)]
        n_exceptions_test[exception_name+'_inc_'+str(i)] = n_exceptions_test[exception_name+'_cnt_'+str(i)] - n_exceptions_test[exception_name+'_cnt_'+str(i + 1)]


n_exceptions_train.to_csv(config['data']['features']['train']['n_exceptions_recent_week_with_diff'], index=False)
n_exceptions_test.to_csv(config['data']['features']['test']['n_exceptions_recent_week_with_diff'], index=False)