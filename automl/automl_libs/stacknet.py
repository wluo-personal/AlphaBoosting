from automl_libs.base_layer_utils import BaseLayerDataRepo, BaseLayerResultsRepo, ModelName
from automl_libs.base_layer_utils import SklearnBLE, LightgbmBLE
from automl_libs.base_layer_utils import compute_layer1_oof, compute_layer2_oof
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
from os import listdir
import pdb
import time
import logging, gc
module_logger = logging.getLogger(__name__)


def layer1(train, test, categorical_cols, feature_cols, label_cols, gs_result_path=''):
    """
    Params:
        train: DataFrame with label
        test: DataFrame without label
        categorical_cols: list of column names
        feature_cols: list of column names
        label_cols: list of column names (multi-classes or single-class)
            e.g. ['label'] or ['label1', 'label2']
    """
    # for debug purpose, read in testY
    import numpy as np
    testY = np.load('/home/kai/data/shiyi/data/flight_data/testY_100k.npy')

    X_train_ordinal = train[feature_cols]  # still df
    y_train = train[label_cols]  # make sure it's df
    X_test_ordinal = test[feature_cols] # still df

    bldr = BaseLayerDataRepo()
    bldr.add_data('flight_data_ordinal', X_train_ordinal, X_test_ordinal, y_train, label_cols, [ModelName.LGB.name])
    # bldr.add_data('flight_data_one_hot', X_train_one_hot, X_test_one_hot, y_train, label_cols,
    #               [ModelName.LOGREG, ModelName.XGB])
    module_logger.debug(bldr)

    gs_res = pd.read_csv('~/data/shiyi/AlphaBoosting/automl/automl_app/project1/output/lgb_grid_search.csv',
                         index_col='Unnamed: 0').sort_values(by=['val_auc'], ascending=False)
    gs_res_dict = gs_res.T.to_dict()
    chosen_res_dict = gs_res.head(4).T.to_dict()  ########################

    # seems need absolute path to save
    oof_path = '/home/kai/data/shiyi/AlphaBoosting/automl/automl_app/project1/output/oof/' # make oof in app.py
    if len(listdir(oof_path)) != 0:
        load_from_file = True
    else:
        load_from_file = False
    base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path,
                                                   load_from_file=load_from_file)
    if load_from_file:
        # remove already processed gs results
        for model_data in base_layer_results_repo.get_model_data_id_list():
            result_index = int(model_data.split('__')[0])
            if chosen_res_dict.pop(result_index, None) is not None:
                module_logger.debug('{} already processed in StackNet, so removed it from chosen_gs_results'.format(result_index))

    model_pool = {}
    if len(chosen_res_dict) > 0:
        for k, v in chosen_res_dict.items():
            lgb_params = v
            module_logger.debug(lgb_params)
            lgb_params['categorical_features'] = categorical_cols
            lgb_params['verbose_eval'] = int(lgb_params['best_round'] / 5)
            lgb_ble = LightgbmBLE(params=lgb_params)
            model_pool[str(k)+'__'+ModelName.LGB.name] = lgb_ble

        layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, model_data_id_list = \
            compute_layer1_oof(bldr, model_pool, label_cols, nfolds=5, sfm_threshold=None)

        for k, v in layer1_est_preds.items():
            print('roc of test', k, roc_auc_score(testY, v))

        base_layer_results_repo.add(layer1_oof_train, layer1_oof_mean_test, layer1_est_preds, model_data_id_list)

        for model_data in model_data_id_list:
            result_index = int(model_data.split('__')[0])
            val_score = gs_res_dict[result_index]['val_auc']
            base_layer_results_repo.add_score(model_data, val_score)

        base_layer_results_repo.save()


def layer2(train, label_cols):
    """
    Params:
        train: DataFrame with label
        label_cols: list of column names (multi-classes or single-class)
            e.g. ['label'] or ['label1', 'label2']

    """
    oof_path = '/home/kai/data/shiyi/AlphaBoosting/automl/automl_app/project1/output/oof/'
    base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path, load_from_file=True)
    print(base_layer_results_repo.show_scores())
    model_pool = {}
    layer2_inputs = {}
    model_name = str(int(time.time()))+'__'+ModelName.LOGREG.name
    model_pool[model_name] = SklearnBLE(LogisticRegression)
    layer2_inputs[model_name] = base_layer_results_repo.get_results(threshold=0.7)
    layer2_est_preds, layer2_oof_train, layer2_oof_test, layer2_model_data_list = \
        compute_layer2_oof(model_pool, layer2_inputs, train, label_cols, 5, 2018)

    # for debug purpose, read in testY
    import numpy as np
    testY = np.load('/home/kai/data/shiyi/data/flight_data/testY_100k.npy')
    for k, v in layer2_est_preds.items():
        print('roc of test', k, roc_auc_score(testY, v))
