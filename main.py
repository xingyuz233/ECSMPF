import os
def fun():
    os.system('export PYTHONPATH="$PYTHONPATH:/data/zhangxingyu/ECSMPF"')
    os.system('python pre/features_n_exceptions.py')
    os.system('python pre/features_n_exceptions_recent_week.py')
    os.system('python pre/features_n_exceptions_recent_week_filter.py')
    os.system('python pre/features_rate_exceptions_recent_week_filter')
    os.system('python pre/features_sum_rate_exceptions_recent_week_filter')
    os.system('python script/train.py --model im_ensemble --feature_type sum_rate_exceptions_recent_week_filter')
fun()