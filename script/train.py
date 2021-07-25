from typing import List
import argparse
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.model import ModelAPI
from src.metrics.metrics import get_metrics
import yaml

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def parse_args() -> object:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgboost', help='the used model')
    parser.add_argument('--feature_type', type=str, default='n_exceptions', help='The used feature type')
    args = parser.parse_args()
    return args


def read_features(feature_type: str):
    feature_path = config['data']['features']['train'][feature_type]
    df_feature = pd.read_csv(feature_path)
    print('Reading features from ...')
    print('Training dataframe: ', df_feature.iloc[:, 3:])
    print('Training labels', df_feature.loc[:, 'nc_down_label'])
    train_x, train_y = df_feature.iloc[:, 3:].values, df_feature.loc[:, 'nc_down_label'].values
    print('Distribution of labels: (label, count)', collections.Counter(train_y).items())
    return train_x, train_y

def main(args):
    # Read features
    train_x, train_y = read_features(args.feature_type)
    print(np.max(train_x), np.min(train_x))
    # Init model
    model = ModelAPI(args.model)
    # Fit model
    model.fit(train_x, train_y)
    # Predict
    hat_y = model.predict(train_x)
    prob_y = model.predict_proba(train_x)
    test_ans = get_metrics(y_true=train_y, y_pred=hat_y, y_score=prob_y)
    print("测试集metrics  acc=%.3f precision=%.3f recall=%.3f \n auc=%.3f balanced_acc=%.3f f1=%.3f \n CM="%(test_ans[0],
    test_ans[1],test_ans[2],test_ans[3],test_ans[4],test_ans[5],test_ans[6]))

if __name__ == '__main__':
    main(parse_args())
    