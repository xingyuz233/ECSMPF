import argparse
import pandas as pd
from tqdm import tqdm
import yaml
config = yaml.load(open('config/config.yaml'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60, help='days before sample_time')
    args = parser.parse_args()
    return args

def features_n_events(days, mode):
    if mode not in ['train', 'test']:
        print('mode must be train or test !')
        raise IndexError
    nc_exceptions = pd.read_csv(config['data']['process'][mode]['nc_exceptions_key_ip_name_value_timelist'])
    nc_sample_label = pd.read_csv(config['data']['raw'][mode]['nc_sample_label'])
    # Convert str to date
    pd.merge()
    str2date = lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
    nc_exceptions['exception_time'] = str2date(nc_exceptions['exception_time'])
    nc_sample_label['sample_time'] = str2date(nc_sample_label['sample_time'])

    new_tuple_list = []
    column_list = ['nc_ip', 'sample_time', 'nc_down_label']
    for i in tqdm(range(nc_sample_label.shape[0]), total=nc_sample_label.shape[0]):
        cur = nc_exceptions.loc[
            (nc_exceptions['nc_ip'] == nc_sample_label.loc[i]['nc_ip']) &
            (nc_exceptions['exception_time'] <= nc_sample_label.loc[i]['sample_time']) &
            (nc_sample_label.loc[i]['sample_time'] - nc_exceptions['exception_time'] <= pd.to_timedelta('60 days'))
        ]
    return None

def main(args):
    features_all_events(args.days)

if __name__ == '__main__':
    main(parse_args())