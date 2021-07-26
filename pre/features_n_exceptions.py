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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60, help='days before sample_time')
    args = parser.parse_args()
    return args


def get_date_cnt_from_intervals(start_date, end_date, date_list):
    if date_list == '': return 0
    return bisect.bisect_left(date_list, end_date) - bisect.bisect_left(date_list, start_date)


def features_n_exceptions(days, mode):
    if mode not in ['train', 'test']:
        print('mode must be train or test !')
        raise IndexError
    
    # Read exceptions and sample data
    print('Read exceptions and sample data')
    nc_exceptions = pd.read_csv(config['data']['process'][mode]['nc_exceptions_key_ip_column_name_value_timelist'])
    nc_sample_label = pd.read_csv(config['data']['raw'][mode]['nc_sample_label'])
    
    # Fill na with empty list for all exception columns
    print('Fill na with empty list for all exception columns')
    for exception_name in exception_name_columns:
        nc_exceptions[exception_name] = nc_exceptions[exception_name].fillna('[]').apply(eval)

    # Join sample table and exception table
    print('Join sample table and exception table')
    nc_sample_label_with_exceptions = nc_sample_label.join(nc_exceptions.set_index('nc_ip'), how='left', on='nc_ip').fillna('')

    # Pre declare some date-util functions
    print('Pre declare some date-util functions')
    str2date = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") if '.' in x else datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    series2date = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
    date2str = lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Compute start time and convert to string
    print('Compute start time and convert to string')
    nc_sample_label_with_exceptions['sample_time'] = series2date(nc_sample_label_with_exceptions['sample_time'])
    nc_sample_label_with_exceptions['start_time'] = nc_sample_label_with_exceptions['sample_time'] - pd.DateOffset(days=days)

    # Compute exception count in recent 'days' days
    print("Compute exception count in recent {} days.".format(days))
    for exception_name in tqdm(exception_name_columns, total=len(exception_name_columns)):
        exception_cnt_name = exception_name + '_cnt' 
        nc_sample_label_with_exceptions[exception_cnt_name] = nc_sample_label_with_exceptions\
            .apply(lambda x: get_date_cnt_from_intervals(date2str(x['start_time']), date2str(x['sample_time']), x[exception_name]), axis=1)
    
    # Save to features
    nc_sample_label_with_exceptions[
        ['nc_ip', 'sample_time', 'nc_down_label'] 
        + [exception_name + '_cnt' for exception_name in exception_name_columns]
    ].to_csv(config['data']['features'][mode]['n_exceptions'], index=False)

def main(args):
    print('Feature process of n_exceptions for TRAIN.')
    features_n_exceptions(args.days, 'train')
    print('Feature process of n_exceptions for TEST.')
    features_n_exceptions(args.days, 'test')

if __name__ == '__main__':
    main(parse_args())