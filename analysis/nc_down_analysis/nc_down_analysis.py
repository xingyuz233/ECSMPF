import argparse
import pandas as pd
import yaml

config = yaml.load(open('config/config.yaml'))
nc_exceptions = pd.read_csv(config.data.process.train.nc_exceptions_key_ip_name_value_timelist)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nc_ip', type=str, default='ab0b5660-b14b-4382-bab1-135f8f64ad64')
    parser.add_argument('--exception_name', type=str, default='exception_name_58')
    parser.add_argument('--sample_date', type=str, default='2016-03-01')
    parser.add_argument('--days', type=int, default=60, help='days before sample_time')
    args = parser.parse_args()
    return args

def exception_num_per_day(nc_ip, exception_name, sample_date, days):
    '''
        计算在sample_date之前的days天里，每天分别有多少个名字为exception_name的异常出现。
        返回长度为days大小的数组。
        比如sample_date='2016-03-01',days=60,则计算'2016-03-01'前60天，每天名为exception_name的异常出现的的个数。
        数据表路径 config.data.process.train.nc_exceptions_key_ip_name_value_timelist
        --------
        Parameters
        --------
        nc_ip: ...
        exception_name: ...
        sample_date: ...
        days: ...
        --------
        Returns
        --------
        events_per_day : list value
    '''
    return None

def main(args):
    exception_num_per_day(*args)

if __name__ == '__main__':
    main(parse_args())