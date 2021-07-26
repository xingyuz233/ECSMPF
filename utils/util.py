import datetime
import os
import yaml
config = yaml.load(open('config/config.yaml'), Loader=yaml.FullLoader)

def make_result_dir(args):
    print("PARAMETER" + "-"*10)
    result_dir = config['result_dir']
    now = datetime.datetime.now()
    S = '{}_{}_{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(args.model.upper(), args.feature_type, now.year, now.month, now.day, now.hour, now.minute, now.second)
    save_dir = os.path.join(result_dir, S)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'parameter.txt'), 'w') as f:
        print("args" + "-"*10)
        for attr, value in sorted(args.__dict__.items()):
            print("args: ", "{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))
        print("config" + "-"*10)
        for attr, value in sorted(config.items()):
            # print("config: ", "{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))
    print("---------" + "-"*10)
    return save_dir