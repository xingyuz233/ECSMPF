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

def make_pseudo_labels():
    
    # Read exceptions and sample data
    print('Read test exceptions and sample data')
    nc_exceptions = pd.read_csv(config['data']['process']['B_test']['nc_exceptions_key_ip_column_name_value_timelist'])
    nc_sample_label = pd.read_csv(config['data']['raw']['B_test']['nc_sample_label'])
    
    # Fill na with empty list for all exception columns
    print('Fill na with empty list for all exception columns')
    for exception_name in exception_name_columns:
        nc_exceptions[exception_name] = nc_exceptions[exception_name].fillna("[]").apply(eval)
    for i in tqdm(range(nc_exceptions.shape[0]), total=nc_exceptions.shape[0]):
        nc_exceptions.loc[i, 'nc_down_time'] = max(nc_exceptions.loc[i, exception_name][-1] for exception_name in exception_name_columns if nc_exceptions.loc[i, exception_name] )
        
    # Pre declare some date-util functions
    print('Pre declare some date-util functions')
    str2date = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") if '.' in x else datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    series2date = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
    date2str = lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Join sample table and exception table
    print('Join sample table and exception table')
    nc_sample_label_with_down_time = nc_sample_label.join(nc_exceptions[['nc_ip', 'nc_down_time']].set_index('nc_ip'), how='left', on='nc_ip').fillna('')
    nc_sample_label_with_down_time['sample_time'] = series2date(nc_sample_label_with_down_time['sample_time'])
    nc_sample_label_with_down_time['nc_down_time'] = series2date(nc_sample_label_with_down_time['nc_down_time'])
    nc_sample_label_with_down_time['down_sample_diff'] = nc_sample_label_with_down_time['nc_down_time'] - nc_sample_label_with_down_time['sample_time']
    nc_sample_label_with_down_time['nc_down_label'] = (nc_sample_label_with_down_time['down_sample_diff'] >= pd.Timedelta(days=0)) & (nc_sample_label_with_down_time['down_sample_diff'] <= pd.Timedelta(days=2))
    
    # Save to process
    nc_sample_label_with_down_time[
        ['nc_ip', 'sample_time', 'nc_down_label']
    ].to_csv(config['data']['process']['B_test']['nc_sample_pseudo_label'], index=False)

def main():
    make_pseudo_labels()

if __name__ == '__main__':
    main()