# ECSMPF 2021 算法赛道

### Packages

* python >= 3.8

```
pandas == 1.2.5
imblearn == 0.7.0
xgboost == 1.3.3
scikit-learn == 0.23.2
```

For other packages, see requirements.txt in detail.

### How to install

```shell
# Before install the requirements of packages, we strongly recommend you to create your own virtualenv (e.g. conda env) in your python environment.  
pip install -r requirements.txt 
```

### Data processing

* Move both 'ecsmpf_round1_train_20210625' and 'ecsmpf_round1_A/B_test_20210625' under 'data/raw' subdirectory.

* Group the exception table by nc_ip, and store all the exception event into a list for each nc_ip and each nc_exception. Then the grouped csv will be outputed into 'data/process' subdirectory.

  ```
  python pre/process_nc_exceptions_csv.py
  ```

###Feature engineering

* Features: the number of all exceptions:

  ```shell
  python pre/features_n_exceptions.py # (1)
  ```

* Features: the number of exceptions in each day of recent one week (based on (1)):

  ```shell
  python pre/features_n_exceptions_recent_week.py # (2)
  # We filter out the samples without any exception in recent one week (based on (2)):
  python pre/features_n_exceptions_recent_week_filter.py # (3)
  ```

* Features: the increasing rate of exceptions in each day of recent one week (based on(1)):

  ```shell
  python pre/features_rate_exceptions_recent_week_filter # (4)
  # Also be filtered out like (3) 
  ```

* Features: the sum of increasing rate (day by day) of exceptions in recent one week 

  ```shell
  python pre/features_sum_rate_exceptions_recent_week_filter # (5)
  # Also be filtered out like (3) 
  ```

### Classifier

* XGBoost + bagging
* LogisticRegression + bagging

### Run

Before training the classifier, make sure to add project root path to PYTHONPATH

```shell
export PYTHONPATH="$PYTHONPATH:[The project root path]"
```

Specify the model (choosen from [im_ensemble, xgboost, logistic_regression]) and feature_type (from (1) to (5) at 'Feature Engineering' section) , train the classifier and output the result. e.g.:

```shell
python scripts/train.py --model im_ensemble --feature_type sum_rate_exceptions_recent_week_filter
```



> `main.py` contains all the scripts from 'data process' to 'run',  you can just run `python main.py` to substitude all steps from 'data process' to 'run'.

> the final submit file is in the 'result' directory.    



### Result

we give some results in the first round for A_test

| 时间                | 特征                                                         | 分类器                           | 训练集混淆矩阵和F1          | 结果F1 |
| ------------------- | ------------------------------------------------------------ | -------------------------------- | --------------------------- | ------ |
| 2021-07-31 22:36:35 | features_sum_rate_exceptions_recent_week_filter (最近7天每天异常+训练集过滤) | XGBOOST(未调参)+IM_ENSEMBLE(1:4) | [[69132   321],[40  1493]]  | 0.2737 |
| 2021-08-01 15:40:25 | features_sum_rate_exceptions_recent_week_filter (最近7天每天异常增长率+训练集过滤) | XGBOOST(未调参)+IM_ENSEMBLE(1:1) | [[67159  2294], [50, 1483]] | 0.1443 |



