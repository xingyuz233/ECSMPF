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
import warnings
warnings.filterwarnings("ignore")

config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def parse_args() -> object:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='im_ensemble', help='the used model')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--feature_type', type=str, default='n_exceptions_recent_week_filter', help='The used feature type')
    args = parser.parse_args()
    return args


def read_train_features(feature_type: str, aug: bool):
    feature_path = config['data']['features']['train'][feature_type]
    df_feature = pd.read_csv(feature_path)
    print('Reading training features from {}.'.format(feature_path))
    if aug:
        aug_feature_path = config['data']['features']['aug_train'][feature_type]
        df_aug_feature = pd.read_csv(aug_feature_path)
        print('Reading aug features from {}.'.format(aug_feature_path))
        print('Union features of train set and aug-train set.')
        df_feature = pd.concat([df_feature, df_aug_feature], axis=0)


    # print('Training dataframe: ', df_feature.iloc[:, 3:])
    # print('Training labels', df_feature.loc[:, 'nc_down_label'])
    train_x, train_y = df_feature.iloc[:, 3:].values, df_feature.loc[:, 'nc_down_label'].astype(int).values
    print('Distribution of labels: (label, count)', collections.Counter(train_y).items())
    return train_x, train_y

def read_test_features(feature_type: str):
    feature_path = config['data']['features']['B_test'][feature_type]
    psudo_label_path = config['data']['process']['B_test']['nc_sample_pseudo_label']
    df_feature = pd.read_csv(feature_path)
    df_pseudo_label = pd.read_csv(psudo_label_path)
    df_feature['nc_down_label'] = df_pseudo_label['nc_down_label']
    print('Reading test features from {}.'.format(feature_path))
    test_x = df_feature.iloc[:, 3:].values
    print('Number of test items: {}.'.format(len(test_x)))
    return test_x, df_feature

def main(args):
    # Read Training features
    train_x, train_y = read_train_features(args.feature_type, args.aug)
    # Init model
    model = ModelAPI(args.model)
    # Fit model
    print(type(train_x), type(train_y))
    model.fit(train_x, train_y)
    # Metrics on train set
    train_hat_y = model.predict(train_x)
    train_prob_y = model.predict_proba(train_x)[:, 1]
    train_ans = get_metrics(y_true=train_y, y_pred=train_hat_y, y_score=train_prob_y)
    # print("Feature importance")
    # print(sorted(zip(config['exception_name']['train'], model.get_feature_importance()), key=lambda x: -x[1]))
    print("metrics on train set: acc=%.3f precision=%.3f recall=%.3f \n auc=%.3f balanced_acc=%.3f f1=%.3f \n CM=%s"\
        %(train_ans[0], train_ans[1],train_ans[2],train_ans[3],train_ans[4],train_ans[5],str(train_ans[6])))
    
    # Predict on test set
    test_x, df_test = read_test_features(args.feature_type)
    test_y = df_test['nc_down_label'].values
    test_hat_y = model.predict(test_x)
    test_prob_y = model.predict_proba(test_x)[:, 1]
    test_ans = get_metrics(y_true=test_y, y_pred=test_hat_y, y_score=test_prob_y)
    print("metrics on test psedo set: acc=%.3f precision=%.3f recall=%.3f \n auc=%.3f balanced_acc=%.3f f1=%.3f \n CM=%s"\
        %(test_ans[0], test_ans[1],test_ans[2],test_ans[3],test_ans[4],test_ans[5],str(test_ans[6])))
    

    df_test['nc_down_label'] = test_hat_y
    df_test['nc_down_score'] = test_prob_y
    # Save result
    result_dir = make_result_dir(args)
    print('Just sort the test cases by their positive scores')
    df_score = df_test.sort_values('nc_down_score', ascending=False)[['nc_ip', 'sample_time', 'nc_down_score']]
    df_score.to_csv(os.path.join(result_dir, 'score.csv'), index=False)
    print('Save the positive predictions by classifier')
    df_result = df_test[df_test['nc_down_label'] == 1][['nc_ip', 'sample_time']]
    print(df_result)
    df_result.to_csv(os.path.join(result_dir, 'result.csv'), index=False, header=False)
        

if __name__ == '__main__':
    main(parse_args())