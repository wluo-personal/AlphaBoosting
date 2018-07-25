import pdb
import os
import sys
import logging
import itertools
import time

nb_dir = os.path.split(os.getcwd())[0]
print(nb_dir)  # e.g.: '/home/kai/data/shiyi/AlphaBoosting/automl'
autolib_dir = '/'.join(nb_dir.split('/')[:-1])
if autolib_dir not in sys.path:
    sys.path.append(autolib_dir)
print(sys.path)

from automl_libs import feature_engineering as fe
from automl_libs import utils
from automl_app import logger_config
from automl_app.app import AlphaBoosting


def get_features_to_gen(function_list, low, high):
    # params = {'split_col': 't', 'coefficient': 10, 'n': 2, 'fillna': 22}
    features_to_gen = []
    for function in function_list:
        for i in range(low, high + 1):
            for combine in itertools.combinations(categorical_features, i):
                if function.__name__ == 'count_std_over_mean':
                    features_to_gen.append(
                        {'params': {'coefficient': 10}, 'function': function, 'feature_cols': list(combine)})
                else:
                    features_to_gen.append({'params': {}, 'function': function, 'feature_cols': list(combine)})

    return features_to_gen


def params_gen(model='lgb'):
    import time
    import numpy as np
    seed = int(time.time() * 1000000) % 45234634
    np.random.seed(seed)
    if model == 'svc':
        params = {
            'C': 0.911  # np.random.rand(),
            #             'metric': 'roc_auc'
        }
    elif model == 'logreg':
        params = {
            'penalty': np.random.choice(['l2']),
            'dual': np.random.choice([True, False]),
            'C': 1,  # np.random.rand(),
            'metric': 'roc_auc'
        }
    elif model == 'lgb':
        params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'num_boost_round': 10,
            'learning_rate': np.random.choice([0.1, 0.03]),  # 0.001]),
            'num_leaves': np.random.choice([15, 31]),  # ,61,127]),
            'num_threads': 8,  # best speed: set to number of real cpu cores, which is vCPU/2
            'max_depth': -1,  # no limit. This is used to deal with over-fitting when #data is small.
            'min_data_in_leaf': np.random.randint(20, 50),  # minimal number of data in one leaf.
            'feature_fraction': np.random.randint(9, 11) / 10,
            'feature_fraction_seed': seed,
            'early_stopping_round': 10,
            'bagging_freq': 1,  # 0 means disable bagging. k: perform bagging at every k iteration
            'bagging_fraction': np.random.randint(4, 11) / 10,  # Randomly select part of data
            'bagging_seed': seed,
            'scale_pos_weight': 1,
            'metric': 'auc'
        }
    elif model == 'xgb':
        params = {
            'eta': np.random.choice([0.01]),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
#             'max_bin': 8,
            'scale_pos_weight': np.random.randint(10,40)/10,
            'seed': seed,
            'nthread': 10,
            'max_depth': np.random.randint(5,12),
            'min_child_weight': np.random.randint(3,50),
            'subsample': np.random.randint(80,101)/100,
            'colsample_bytree': np.random.randint(10,35)/100,
#             'colsample_bylevel': 0.5,
#             'alpha': 0,
            'lambda': np.random.randint(5, 15)/10,
#             'gamma': 0,
            'num_boost_round': 10000,
            'early_stopping_rounds': 200
        }
    elif model == 'catb':
        params = {
            'iterations': 12000,
            'depth': np.random.randint(4, 10),
            'l2_leaf_reg': np.random.randint(0, 31) / 10,
            #     'custom_metric': 'AUC',
            'eval_metric': 'AUC',
            'random_seed': seed,
            # 'use_best_model': True,  # comment this if not doing cv
            'logging_level': 'Verbose',
            'thread_count': 15
        }
    elif model == 'nn':
        params = {
            'nn_seed': int(time.time() * 1000000) % 45234634,
            'ep_for_lr': np.random.randint(2, 10),
            'lr_init': 0.01,
            'lr_fin': np.random.randint(1, 5) / 1000,  # if == lr_init, then no lr decay
            'batch_size': np.random.choice([128, 256, 512, 1024]),
            "pred_batch_size": 5000,
            'max_ep': 100,
            'patience': 10,  # np.random.randint(10, 25),
            'cat_emb_outdim': 30,  # could be a constant or a dict (col name:embed out dim). e.g.:
            # embed_outdim = [3, 3, 8, 8, 3]
            # embed_outdim_dict = dict(zip(X_train.columns.values, embed_outdim))
            # then assige embed_outdim_dict to cat_emb_outdim
            'cat_emb_drop_rate': np.random.randint(1, 4) / 10,
            'num_layers_dense_units': [1000, 500, 100],
            'num_layers_drop_rate': np.random.randint(2, 6) / 10,
            'combined_layers_dense_units': [100, 50],
            'combined_layers_drop_rate': np.random.randint(1, 3) / 10,
            'monitor': 'val_auc',  # or val_loss (MUST HAVE)
            'mode': 'max',  # MUST HAVE
            'int_list': ['num_layers_dense_units', 'combined_layers_dense_units']
        }
    elif model == 'stacknet_layer2_nn':
        params = {
            'nn_seed': int(time.time() * 1000000) % 45234634,
            'ep_for_lr': 1,
            'lr_init': 0.01,
            'lr_fin': 0.01,  # if == lr_init, then no lr decay
            'batch_size': 128,
            "pred_batch_size": 50000,
            'best_epoch': 1,
            'patience': 1,
            'categorical_feature': [],
            'cat_emb_outdim': 50,   # could be a constant or a dict (col name:embed out dim). e.g.:
                                    # embed_outdim = [3, 3, 8, 8, 3]
                                    # embed_outdim_dict = dict(zip(X_train.columns.values, embed_outdim))
            'num_layers_dense_units': [],
            'combined_layers_dense_units': [10, 5],
            'combined_layers_drop_rate': 0,
            'monitor': 'val_auc',  # or val_loss (MUST HAVE)
            'mode': 'max',  # MUST HAVE
            'int_list': ['num_layers_dense_units', 'combined_layers_dense_units']
        }
    return params, seed


class Logger(object):
    def __init__(self, logtofile=True, logfilename='log'):
        self.terminal = sys.stdout
        self.logfile = "{}_{}.log".format(logfilename, int(time.time()))
        self.logtofile = logtofile

    def write(self, message):
        #         self.terminal.write(message)
        if self.logtofile:
            self.log = open(self.logfile, "a")
            self.log.write('[' + utils.get_time() + '] ' + message)
            self.log.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def divert_printout_to_file():
    sys.stdout = Logger(logfilename='logfile')


def kaggle_auto_sub(npyfile):
    import os
    import numpy as np
    import pandas as pd
    pred = np.load(npyfile + '.npy')
    path = '/home/kai/data/shiyi/AlphaBoosting/automl/automl_app/project_home_credit/'
    sub = pd.read_csv(path + 'data/sample_submission.csv')
    sub['TARGET'] = pred
    filename = path + 'subs/' + npyfile.split('/')[-1] + '.csv.gz'
    sub.to_csv(filename, index=False, compression='gzip')
    cmd = 'kaggle competitions submit -c home-credit-default-risk -f ' + filename + ' -m "auto submitted"'
    os.system(cmd)


if __name__ == '__main__':

    # seems not needed to divert if using nohup python main.py &
    # divert_printout_to_file()  # note: comment this to use pdb

    categorical_features = ['blabla']  # does not matter since not using the fe library.
    # features_to_gen = get_features_to_gen([fe.count, fe.unique_count,
    #                                        fe.cumulative_count,
    #                                        fe.reverse_cumulative_count,
    #                                        fe.variance, fe.count_std_over_mean],
    #                                       2, 2)# len(categorical_features))
    # features_to_gen = get_features_to_gen([fe.count, fe.unique_count, fe.cumulative_count])
    features_to_gen = []  #get_features_to_gen([fe.count], 2, 2)  # len(categorical_features))
    print(len(features_to_gen))
    print()
    print(features_to_gen)

    project_path = './'
    logger_config.config(project_path + 'project.log', file_loglevel=logging.INFO)
    automl_config_file = project_path + 'automl_config.json'
    run_record_file_name = project_path + 'last_run_record.json'  # don't created this file
    AlphaBoosting(automl_config_file, features_to_gen, params_gen, kaggle_auto_sub)