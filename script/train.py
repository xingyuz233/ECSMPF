from typing import List
import argparse
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.model import ModelAPI
from src.metrics.metrics import get_metrics
from utils.util import *
import yaml

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def parse_args() -> object:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='im_ensemble', help='the used model')
    parser.add_argument('--feature_type', type=str, default='n_exceptions_recent_week', help='The used feature type')
    parser.add_argument('--topk', type=str, default=None)
    args = parser.parse_args()
    return args


def read_train_features(feature_type: str):
    feature_path = config['data']['features']['train'][feature_type]
    df_feature = pd.read_csv(feature_path)
    print('Reading training features from {}.'.format(feature_path))
    # print('Training dataframe: ', df_feature.iloc[:, 3:])
    # print('Training labels', df_feature.loc[:, 'nc_down_label'])
    train_x, train_y = df_feature.iloc[:, 3:].values, df_feature.loc[:, 'nc_down_label'].astype(int).values
    print('Distribution of labels: (label, count)', collections.Counter(train_y).items())
    return train_x, train_y

def read_test_features(feature_type: str):
    feature_path = config['data']['features']['test'][feature_type]
    df_feature = pd.read_csv(feature_path)
    print('Reading test features from {}.'.format(feature_path))
    test_x = df_feature.iloc[:, 3:].values
    print('Number of test items: {}.'.format(len(test_x)))
    return test_x, df_feature

def main(args):
    # Read Training features
    
    train_x, train_y = read_train_features(args.feature_type)
    # Init model
    model = ModelAPI(args.model)
    # Fit model
    model.fit(train_x, train_y)
    # Metrics on train set
    hat_y = model.predict(train_x)
    prob_y = model.predict_proba(train_x)[:, 1]
    test_ans = get_metrics(y_true=train_y, y_pred=hat_y, y_score=prob_y)
    # print("Feature importance")
    # print(sorted(zip(config['exception_name']['train'], model.get_feature_importance()), key=lambda x: -x[1]))
    print("metrics  acc=%.3f precision=%.3f recall=%.3f \n auc=%.3f balanced_acc=%.3f f1=%.3f \n CM=%s"\
        %(test_ans[0], test_ans[1],test_ans[2],test_ans[3],test_ans[4],test_ans[5],str(test_ans[6])))
    
    # Predict on test set
    test_x, df_test = read_test_features(args.feature_type)
    hat_y = model.predict(test_x)
    score_y = model.predict_proba(test_x)[:, 1]
    df_test['nc_down_label'] = hat_y
    df_test['nc_down_score'] = score_y
    # Save result
    result_dir = make_result_dir(args)
    if args.topk:
        print('Return topk of the highest scores predicted by the classifier')
        df_result = df_test.sort_values('nc_down_score', ascending=False).head(args.topk)[['nc_ip', 'sample_time']]
        print(df_result)
        df_result.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
    else:
        print('Do not set topk, so just save the positive predictions by classifier')
        df_result = df_test[df_test['nc_down_label'] == 1][['nc_ip', 'sample_time']]
        print(df_result)
        df_result.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
        

if __name__ == '__main__':
    main(parse_args())
    