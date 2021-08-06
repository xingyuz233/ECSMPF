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
n_exceptions_test = pd.read_csv(config['data']['features']['B_test']['n_exceptions_recent_week'])

n_exceptions_train_filter = n_exceptions_train[sum([n_exceptions_train[exception_name+'_cnt_'+str(i)] for exception_name in exception_name_columns for i in range(0, 8)]) > 0]
    
n_exceptions_train_filter.to_csv(config['data']['features']['train']['n_exceptions_recent_week_filter'], index=False)
n_exceptions_test.to_csv(config['data']['features']['B_test']['n_exceptions_recent_week_filter'], index=False)