from automl_libs import BaseLayerDataRepo, BaseLayerResultsRepo, ModelName
from automl_libs import SklearnBLE, LightgbmBLE, NNBLE
from automl_libs import compute_layer1_oof, compute_layer2_oof
from automl_libs import utils
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from os import listdir
import pdb
import time
import logging, gc
module_logger = logging.getLogger(__name__)


def layer1(train, test, categorical_cols, feature_cols, label_cols, top_n_gs,
           oof_nfolds, oof_path, metric, gs_result_path=''):
    """
    Params:
        train: DataFrame with label
        test: DataFrame without label
        categorical_cols: list of column names
        feature_cols: list of column names
        label_cols: list of column names (multi-classes or single-class) (required by BaseLayerDataRepo)
            e.g. ['label'] or ['label1', 'label2']
        top_n_gs: int. choose top N from grid search results of each model
    """
    X_train_ordinal = train[feature_cols]  # still df
    y_train = train[label_cols]  # make sure it's df
    X_test_ordinal = test[feature_cols] # still df

    bldr = BaseLayerDataRepo()
    bldr.add_data('flight_data_ordinal', X_train_ordinal, X_test_ordinal, y_train, label_cols,
                  [ModelName.LGB.name, ModelName.NN.name])
    # bldr.add_data('flight_data_one_hot', X_train_one_hot, X_test_one_hot, y_train, label_cols,
    #               [ModelName.LOGREG, ModelName.XGB])
    module_logger.debug(bldr)

    metrics_callback = _get_metrics_callback(metric)

    # may be search gs result path and find all result files?
    for filename in listdir(gs_result_path):
        if '_grid_search' in filename:
            model_type = filename.split('_')[0]  # LGB, NN, etc...
            gs_res = pd.read_csv(gs_result_path+filename, index_col='Unnamed: 0').sort_values(by=['val_auc'], ascending=False)
            gs_res_dict = gs_res.T.to_dict()
            chosen_res_dict = gs_res.head(top_n_gs).T.to_dict()

            if len(listdir(oof_path)) != 0:
                load_from_file = True
            else:
                load_from_file = False
            base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path,
                                                           load_from_file=load_from_file)
            if load_from_file:
                # remove already processed gs results
                for model_data in base_layer_results_repo.get_model_data_id_list():
                    result_index = model_data.split('__')[0]
                    if chosen_res_dict.pop(result_index, None) is not None:
                        module_logger.info('{} already processed in StackNet, so removed it from chosen_gs_results'.format(result_index))

            model_pool = {}
            if len(chosen_res_dict) > 0:
                for k, v in chosen_res_dict.items():
                    params = v
                    module_logger.info('using {} params: {} to do oof'.format(model_type, k))
                    module_logger.debug(params)
                    params['categorical_feature'] = categorical_cols
                    if model_type == 'lgb':
                        params['verbose_eval'] = int(params['best_round'] / 5)
                        base_layer_estimator = LightgbmBLE(params=params)
                        model_pool[str(k)+'__'+ModelName.LGB.name] = base_layer_estimator
                    elif model_type == 'nn':
                        params['verbose_eval'] = 1
                        base_layer_estimator = NNBLE(params=params)
                        model_pool[str(k)+'__'+ModelName.NN.name] = base_layer_estimator

                layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, layer1_cv_score, model_data_id_list = \
                    compute_layer1_oof(bldr, model_pool, label_cols, nfolds=oof_nfolds,
                                       sfm_threshold=None, metrics_callback=metrics_callback)

                base_layer_results_repo.add(layer1_oof_train, layer1_oof_mean_test, layer1_est_preds,
                                            layer1_cv_score, model_data_id_list)

                # the following add/update scores to model_data in the repo, so it
                # needs to be done after the [add] function, which stores
                # model_data into the repo.
                for model_data in model_data_id_list:
                    result_index = model_data.split('__')[0]
                    val_score = gs_res_dict[result_index]['val_'+metric]
                    base_layer_results_repo.add_score(model_data, val_score)
                    base_layer_results_repo.update_report(model_data, 'gs_val_{}'.format(metric), val_score)

                # for debug purpose, read in testY  # TODO, remove after development
                import numpy as np
                testY = np.load('/home/kai/data/shiyi/data/flight_data/testY_100k.npy')
                for k, v in layer1_est_preds.items():
                    base_layer_results_repo.update_report(k, 'test_score', roc_auc_score(testY, v))

                base_layer_results_repo.save()


def layer2(train, label_cols, oof_path, metric, save_report):
    """
    Params:
        train: DataFrame with label
        label_cols: list of column names (multi-classes or single-class)
            e.g. ['label'] or ['label1', 'label2']

    """
    metrics_callback = _get_metrics_callback(metric)
    base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path, load_from_file=True)
    module_logger.info('All available layer1 model_data:')
    module_logger.info(base_layer_results_repo.show_scores())
    model_pool = {}
    layer2_inputs = {}
    layer2_chosen_model_data = {}

    model_id = utils.get_random_string()
    model_name = model_id+'__'+ModelName.LOGREG.name
    model_pool[model_name] = SklearnBLE(LogisticRegression)
    chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds, chosen_model_data_list = \
        base_layer_results_repo.get_results(layer='layer1', threshold=0.70)
    layer2_inputs[model_name] = chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds
    layer2_chosen_model_data[model_id] = ' | '.join(['_'.join(name.split('_')[:3]) for name in chosen_model_data_list])

    # model_id = utils.get_random_string()
    # model_name = model_id+'__'+ModelName.LOGREG.name
    # model_pool[model_name] = SklearnBLE(DecisionTreeClassifier)
    # chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds, chosen_model_data_list = \
    #     base_layer_results_repo.get_results(layer='layer1', threshold=0.713)
    # layer2_inputs[model_name] = chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds
    # layer2_chosen_model_data[model_id] = ' | '.join(['_'.join(name.split('_')[:3]) for name in chosen_model_data_list])

    layer2_est_preds, layer2_oof_train, layer2_oof_test, layer2_cv_score, layer2_model_data_list = \
        compute_layer2_oof(model_pool, layer2_inputs, train, label_cols, 5, 2018, metric=metric, metrics_callback=metrics_callback)

    base_layer_results_repo.add(layer2_oof_train, layer2_oof_test, layer2_est_preds,
                                layer2_cv_score, layer2_model_data_list)

    for model_data in layer2_model_data_list:
        model_id = model_data.split('__')[0]
        base_layer_results_repo.add_score(model_data, layer2_cv_score[model_data])
        base_layer_results_repo.update_report(model_data, 'chosen model_data', layer2_chosen_model_data[model_id])

    # for debug purpose, read in testY  # TODO, remove after development
    import numpy as np
    testY = np.load('/home/kai/data/shiyi/data/flight_data/testY_100k.npy')
    for k, v in layer2_est_preds.items():
        base_layer_results_repo.update_report(k, 'test_score', roc_auc_score(testY, v))

    base_layer_results_repo.save()

    if save_report:
        stacknet_report = base_layer_results_repo.get_report()
        stacknet_report_file = oof_path + 'stacknet_report.csv'
        stacknet_report.to_csv(stacknet_report_file, index=False)
        module_logger.info('StackNet Report saved at {}'.format(stacknet_report_file))


def _get_metrics_callback(metric):
    metrics_callback = None
    if metric == 'auc':
        metrics_callback = roc_auc_score
    else:
        module_logger.warning('metric is NOT auc, metric callback will be None')
    return metrics_callback
