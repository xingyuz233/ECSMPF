import pandas as pd
import collections
from tqdm import tqdm
import csv
import os
import yaml
config = yaml.load(open('config/config.yaml'))

exception_raw_train_path = config['data']['raw']['train']['nc_exceptions']
exception_raw_test_path = config['data']['raw']['test']['nc_exceptions']
exception_save_train_path_1 = config['data']['process']['train']['nc_exceptions_key_ip_name_value_timelist']
exception_save_train_path_2 = config['data']['process']['train']['nc_exceptions_key_ip_column_name_value_timelist']
exception_save_test_path_1 = config['data']['process']['test']['nc_exceptions_key_ip_name_value_timelist']
exception_save_test_path_2 = config['data']['process']['test']['nc_exceptions_key_ip_column_name_value_timelist']
exception_name_train = config['exception_name']['train']

def process(load_path, save_path_1, save_path_2):
    # 从load_path中读入exception, 由于文件非常大, 为方便后续处理, 用csv将其读入
    nc_ip_exception_info = collections.defaultdict(list)
    with open(load_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            nc_ip_exception_info[(row['nc_ip'], row['exception_name'])].append(row['exception_time'])
    # 根据Sample time 排序
    for key in tqdm(nc_ip_exception_info.keys(), total=len(nc_ip_exception_info)):
        nc_ip_exception_info[key].sort()
    # 制作第一个表
    tuples = []
    for key in tqdm(nc_ip_exception_info.keys(), total=len(nc_ip_exception_info)):
        tuples.append((key[0], key[1], nc_ip_exception_info[key]))
    res_tbl = pd.DataFrame(tuples, columns=['nc_ip', 'nc_exception_name', 'nc_exception_time_info'])
    del tuples
    res_tbl.to_csv(save_path_1, encoding='utf-8', index=False)
    del res_tbl

    # 制作第二个表
    row_names = list(set([key[0] for key in nc_ip_exception_info.keys()]))
    # column_names = sorted(list(set([key[1] for key in nc_ip_exception_info.keys()])))
    column_names = exception_name_train
    new_tuples = []
    for row in row_names:
        new_tuples.append([row] + [nc_ip_exception_info[(row, col)] if nc_ip_exception_info[(row, col)] else None for col in column_names])
    new_tbl = pd.DataFrame(new_tuples, columns=['nc_ip'] + column_names)
    del new_tuples
    new_tbl.to_csv(save_path_2, encoding='utf-8', index=False)
    del new_tbl

if __name__ == '__main__':
    # process(exception_raw_test_path, exception_save_test_path_1, exception_save_test_path_2)
    process(exception_raw_train_path, exception_save_train_path_1, exception_save_train_path_2)
