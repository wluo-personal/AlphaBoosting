from automl_libs import BaseLayerDataRepo, BaseLayerResultsRepo, ModelName
from automl_libs import SklearnBLE, LightgbmBLE, NNBLE, XgboostBLE, CatBoostBLE
from automl_libs import compute_layer1_oof, compute_layer2andmore_oof
from automl_libs import utils
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd
from os import listdir
import pdb
import time
import os
import ast
import logging, gc
module_logger = logging.getLogger(__name__)


def layer1(data_name, train, test, y_test, categorical_cols, feature_cols, label_cols, params_source,
           params_gen, layer1_models, build_layer1_amount, top_n_gs, top_n_by, oof_nfolds, stratified,
           seed, oof_path, metric, ascending, auto_sub_func, preds_save_path, pg_save_path, gs_result_path=''):
    """
    :param params_source: (str) either 'gs' (grid search, meaning params retrieved from saved gs result csv),
        or 'pg' (param_gen, meaning params retrieved from the param_gen function)
    :param pg_save_path: (str) if params_source == 'pg', then stacknet will build layer1 using params from params_gen. The
        params and results will be saved in this path. e.g. './output/'
    :param data_name: data name. e.g. 'ordinal', 'one_hot', etc
    :param train: DataFrame with label
    :param test: DataFrame without label
    :param y_test: array-like. In Kaggle, we don't know it.
    :param categorical_cols: list of column names
    :param feature_cols: list of column names
    :param label_cols: list of column names (multi-classes or single-class) (required by BaseLayerDataRepo)
                e.g. ['label'] or ['label1', 'label2']
    :param top_n_gs: int. choose top N from EACH grid search results (combination of model and data)
    :param oof_nfolds: nfolds in oof
    :param oof_path: path to save and load oof
    :param metric: usually 'auc' for binary classification
    :param gs_result_path: path to load grid search result
    :return: None
    """
    if not os.path.exists(preds_save_path):
        os.makedirs(preds_save_path)

    X_train = train[feature_cols]  # still df
    y_train = train[label_cols]  # make sure it's df
    X_test = test[feature_cols] # still df

    bldr = BaseLayerDataRepo()
    # leave the compatible model list empty
    bldr.add_data(data_name, X_train, X_test, y_train, label_cols, [])
    module_logger.debug(bldr)

    metrics_callback = _get_metrics_callback(metric)

    if len(listdir(oof_path)) != 0:
        load_from_file = True
    else:
        load_from_file = False
    base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path,
                                                   load_from_file=load_from_file)

    # may be search gs result path and find all result files?
    if params_source == 'gs':
        if layer1_models is None:
            layer1_models = ['xgb', 'lgb', 'catb', 'nn']
        for filename in listdir(gs_result_path):
            if '_grid_search' in filename and '.csv' in filename and data_name in filename:
                model_type = filename.split('_')[0]  # LGB, NN, etc...
                if model_type not in layer1_models:
                    continue

                # get the val_metric col name
                gs_res = pd.read_csv(gs_result_path + filename, index_col='Unnamed: 0')
                for col in gs_res.columns:
                    if top_n_by in col.lower() and 'val' in col.lower():
                        val_metric_colname = col
                        module_logger.debug('Topn will be selected based on [{}]'.format(val_metric_colname))
                        break
                if model_type == 'nn':
                    if top_n_by == 'logloss':
                        val_metric_colname = 'val_loss'
                    elif top_n_by == 'auc':
                        val_metric_colname = 'val_auc'
                gs_res = gs_res.sort_values(by=val_metric_colname, ascending=ascending)
                # try:
                #     # in grid search result of catboost, it's 'val_AUC',
                #     # in all other models, it's 'val_auc'
                #     gs_res = pd.read_csv(gs_result_path+filename, index_col='Unnamed: 0')\
                #         .sort_values(by='val_'+top_n_by.lower(), ascending=ascending)
                # except KeyError:
                #     gs_res = pd.read_csv(gs_result_path+filename, index_col='Unnamed: 0') \
                #         .sort_values(by='val_'+top_n_by.upper(), ascending=ascending)
                # gs_res_dict = gs_res.T.to_dict()
                chosen_res_dict = gs_res.head(top_n_gs).T.to_dict()

                if load_from_file:
                    # remove already processed gs results
                    for model_data in base_layer_results_repo.get_model_data_id_list():
                        result_index = model_data.split('__')[0]
                        if chosen_res_dict.pop(result_index, None) is not None:
                            module_logger.info('{} already processed in StackNet, so removed it from chosen_gs_results'
                                               .format(result_index))

                # model_pool = {}  # move this inside the for loop so that saving is performed per model
                if len(chosen_res_dict) > 0:
                    for k, v in chosen_res_dict.items():
                        model_pool = {}  # since it's now inside the loop, it will only contain one model each iteration
                        params = v
                        module_logger.info('using {} params: {} to do oof'.format(model_type, k))
                        module_logger.debug(params)

                        # [50, 20] is converted to "[50, 20]" after saved in csv
                        # so convert them back to [50, 20]
                        if 'int_list' in params:
                            for int_list_name in ast.literal_eval(ast.literal_eval(params['int_list'])):
                                params[int_list_name] = ast.literal_eval(ast.literal_eval(params[int_list_name]))

                        params['categorical_feature'] = categorical_cols
                        if model_type == 'lgb':
                            params['verbose_eval'] = int(params['best_round'] / 10)
                            base_layer_estimator = LightgbmBLE(params=params)
                            model_pool[str(k)+'__'+ModelName.LGB.name] = base_layer_estimator
                            bldr.add_compatible_model(data_name, ModelName.LGB.name)
                        elif model_type == 'xgb':
                            params['verbose_eval'] = int(params['best_round'] / 10)
                            base_layer_estimator = XgboostBLE(params=params)
                            model_pool[str(k)+'__'+ModelName.XGB.name] = base_layer_estimator
                            bldr.add_compatible_model(data_name, ModelName.XGB.name)
                        elif model_type == 'catb':
                            original_params_example, _ = params_gen('catb')
                            keys_in_original_params = original_params_example.keys()
                            model_params = {}
                            for key in keys_in_original_params:
                                model_params[key] = params[key]
                            # if it's true, then the stacknet will stick to the best_round computed by grid search
                            # hence use_best_model will be set to true
                            if params['use_best_model']:
                                model_params['iterations'] = params['best_round']
                                model_params['use_best_model'] = False
                            print(model_params)
                            model_params['categorical_feature'] = categorical_cols
                            model_params['verbose_eval'] = int(params['iterations'] / 10)
                            base_layer_estimator = CatBoostBLE(params=model_params)
                            model_pool[str(k)+'__'+ModelName.CATB.name] = base_layer_estimator
                            bldr.add_compatible_model(data_name, ModelName.CATB.name)
                        elif model_type == 'nn':
                            params['verbose_eval'] = 1
                            base_layer_estimator = NNBLE(params=params)
                            model_pool[str(k)+'__'+ModelName.NN.name] = base_layer_estimator
                            bldr.add_compatible_model(data_name, ModelName.NN.name)

                        # the following was outside the for loop, now they are inside
                        # so that we can save after each model is done. (because a model
                        # might need hours of finish computing oof, we want to save it
                        # once it's done)

                        if len(model_pool) > 0:
                            layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, layer1_cv_score, \
                            layer1_cv_train_score, model_data_id_list = \
                                compute_layer1_oof(bldr, model_pool, label_cols, auto_sub_func, preds_save_path,
                                                   nfolds=oof_nfolds, stratified=stratified, seed=seed,
                                                   sfm_threshold=None, metrics_callback=metrics_callback)

                            base_layer_results_repo.add(layer1_oof_train, layer1_oof_mean_test, layer1_est_preds,
                                                        layer1_cv_score, model_data_id_list)

                            # the following add/update scores to model_data in the repo, so it
                            # needs to be done after the [add] function, which stores
                            # model_data into the repo.
                            for model_data in model_data_id_list:
                                result_index = model_data.split('__')[0]
                                val_score = chosen_res_dict[result_index][val_metric_colname]
                                # try:
                                #     # in grid search result of catboost, it's val_AUC,
                                #     # in all others, it's val_auc
                                #     val_score = chosen_res_dict[result_index]['val_' + metric.lower()]
                                # except KeyError:
                                #     val_score = chosen_res_dict[result_index]['val_' + metric.upper()]
                                # base_layer_results_repo.add_score(model_data, val_score)  # will use oof_cv_score instead
                                base_layer_results_repo.update_report(model_data, 'gs_val_{}'.format(metric), val_score)

                            for k, v in layer1_est_preds.items():  # there is only one in layer1_est_preds (see the comment above)
                                base_layer_results_repo.update_report(k, 'nfolds', oof_nfolds)
                                base_layer_results_repo.update_report(k, 'seed', seed)
                                if y_test is not None:
                                    base_layer_results_repo.update_report(k, 'test_score', metrics_callback(y_test, v))

                            base_layer_results_repo.save()

    elif params_source == 'pg':
        for model_type in layer1_models:
            for i in range(build_layer1_amount):
                module_logger.info('Build Layer1 {}: {} of {}'.format(model_type, i+1, build_layer1_amount))
                run_id = str(utils.get_random_string())
                params, seed = params_gen(model_type)
                module_logger.info('Params: {}'.format(params))
                module_logger.info('using {} randomly generated params to do oof {}'.format(model_type, run_id))
                module_logger.debug(params)
                model_pool = {}  # since it's now inside the loop, it will only contain one model each iteration
                params['categorical_feature'] = categorical_cols
                if model_type == 'lgb':
                    base_layer_estimator = LightgbmBLE(params=params)
                    model_pool[run_id + '__' + ModelName.LGB.name] = base_layer_estimator
                    bldr.add_compatible_model(data_name, ModelName.LGB.name)
                elif model_type == 'xgb':
                    base_layer_estimator = XgboostBLE(params=params)
                    model_pool[run_id + '__' + ModelName.XGB.name] = base_layer_estimator
                    bldr.add_compatible_model(data_name, ModelName.XGB.name)
                elif model_type == 'catb':
                    params['categorical_feature'] = categorical_cols
                    base_layer_estimator = CatBoostBLE(params=params)
                    model_pool[run_id + '__' + ModelName.CATB.name] = base_layer_estimator
                    bldr.add_compatible_model(data_name, ModelName.CATB.name)
                elif model_type == 'nn':
                    base_layer_estimator = NNBLE(params=params)
                    model_pool[run_id + '__' + ModelName.NN.name] = base_layer_estimator
                    bldr.add_compatible_model(data_name, ModelName.NN.name)

                if len(model_pool) > 0:
                    # the following was outside the for loop, now they are inside
                    # so that we can save after each model is done. (because a model
                    # might need hours of finish computing oof, we want to save it
                    # once it's done)
                    layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, layer1_cv_score, \
                    layer1_cv_train_score, model_data_id_list = \
                        compute_layer1_oof(bldr, model_pool, label_cols, auto_sub_func, preds_save_path,
                                           nfolds=oof_nfolds, stratified=stratified, seed=seed,
                                           sfm_threshold=None, metrics_callback=metrics_callback)

                    base_layer_results_repo.add(layer1_oof_train, layer1_oof_mean_test, layer1_est_preds,
                                                layer1_cv_score, model_data_id_list)

                    for k, v in layer1_est_preds.items():  # there is only one in layer1_est_preds (see the comment above)
                        base_layer_results_repo.update_report(k, 'chosen model_data', 'N/A')
                        base_layer_results_repo.update_report(k, 'nfolds', oof_nfolds)
                        base_layer_results_repo.update_report(k, 'seed', seed)
                        base_layer_results_repo.update_report(k, 'stratified', stratified)
                        if y_test is not None:
                            base_layer_results_repo.update_report(k, 'test_score', metrics_callback(y_test, v))

                    base_layer_results_repo.save()

                    params['oof_seed'] = seed
                    params['cv_val_'+metric] = list(layer1_cv_score.values())[0]
                    params['cv_train_'+metric] = list(layer1_cv_train_score.values())[0]
                    utils.save_params_and_result(run_id, model_type, data_name, 'stacknet', params,
                                                 ['categorical_column', 'verbose'], pg_save_path)


def layer2andmore(train, y_test, label_cols, params_gen, oof_path, oof_nfolds, stratified, seed, metric,
                  layer1_thresh_or_chosen, layer, layer2andmore_models, auto_sub_func, preds_save_path):
    """
    :param layer: str. 'layer2', 'layer3', etc..
    :param train: DataFrame with label
    :param y_test: array-like. In Kaggle, we don't know it.
    :param label_cols: list of column names (multi-classes or single-class)
                e.g. ['label'] or ['label1', 'label2']
    :param params_gen: needed for models like: NN
    :param oof_path: path of save and load oof
    :param metric: usually 'auc'.
    :param layer1_thresh_or_chosen: either float (0,1) as threshold, or list of chosen model_data
    :param layer2andmore_models: currently support models: logreg, nn
    :return: None
    """
    if not os.path.exists(preds_save_path):
        os.makedirs(preds_save_path)

    metrics_callback = _get_metrics_callback(metric)
    base_layer_results_repo = BaseLayerResultsRepo(label_cols=label_cols, filepath=oof_path, load_from_file=True)
    module_logger.info('All available model_data:')
    module_logger.info(base_layer_results_repo.show_scores())
    model_pool = {}
    layer2andmore_inputs = {}
    layer2andmore_chosen_model_data = {}

    if isinstance(layer1_thresh_or_chosen, float) and 0 < layer1_thresh_or_chosen < 1:
        chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds, chosen_model_data_list = \
            base_layer_results_repo.get_results(chosen_from_layer='layer1', threshold=layer1_thresh_or_chosen)
    elif isinstance(layer1_thresh_or_chosen, list):
        chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds, chosen_model_data_list = \
            base_layer_results_repo.get_results(chosen_ones=layer1_thresh_or_chosen)
    else:
        raise ValueError('layer1_thresh_or_chosen has unacceptable type {}. please pass in a float(0,1) or a list'
                         .format(type(layer1_thresh_or_chosen)))

    for layer2andmore_model in layer2andmore_models:
        if layer2andmore_model == 'logreg':
            model_id = utils.get_random_string()
            model_name = model_id+'__'+ModelName.LOGREG.name
            model_pool[model_name] = SklearnBLE(LogisticRegression)
        elif layer2andmore_model == 'et':
            model_id = utils.get_random_string()
            model_name = model_id+'__'+ModelName.ET.name
            model_pool[model_name] = SklearnBLE(ExtraTreeClassifier)
        elif layer2andmore_model == 'rf':
            model_id = utils.get_random_string()
            model_name = model_id+'__'+ModelName.RF.name
            model_pool[model_name] = SklearnBLE(RandomForestClassifier)
        elif layer2andmore_model == 'lgb':
            model_id = utils.get_random_string()
            model_name = model_id+'__'+ModelName.LGB.name
            model_pool[model_name] = SklearnBLE(clf=LGBMClassifier, params={'n_jobs': 4})
        elif layer2andmore_model == 'xgb':
            model_id = utils.get_random_string()
            model_name = model_id+'__'+ModelName.XGB.name
            model_pool[model_name] = SklearnBLE(clf=XGBClassifier, params={'n_jobs': 4})
        elif layer2andmore_model == 'nn':
            model_id = utils.get_random_string()
            nn_model_param, _ = params_gen('stacknet_layer2_nn')
            model_name = model_id+'__'+ModelName.NN.name
            model_pool[model_name] = NNBLE(params=nn_model_param)
        else:
            raise ValueError('{} is not supported in layer2andmore'.format(layer2andmore_model))
        layer2andmore_inputs[model_name] = chosen_layer_oof_train, chosen_layer_oof_test, chosen_layer_est_preds
        layer2andmore_chosen_model_data[model_id] = ' | '.join(['_'.join(name.split('_')[:3]) for name in chosen_model_data_list])

    # decision tree: bad performance
    # ...
    # model_pool[model_name] = SklearnBLE(DecisionTreeClassifier)
    # ...


    layer2andmore_est_preds, layer2andmore_oof_train, layer2andmore_oof_test, layer2andmore_cv_score, \
    layer2andmore_model_data_list = compute_layer2andmore_oof(
        model_pool, layer, layer2andmore_inputs, train, label_cols, oof_nfolds, stratified, seed, auto_sub_func,
        preds_save_path, metric=metric, metrics_callback=metrics_callback)

    base_layer_results_repo.add(layer2andmore_oof_train, layer2andmore_oof_test, layer2andmore_est_preds,
                                layer2andmore_cv_score, layer2andmore_model_data_list)

    for model_data in layer2andmore_model_data_list:
        model_id = model_data.split('__')[0]
        base_layer_results_repo.add_score(model_data, layer2andmore_cv_score[model_data])
        base_layer_results_repo.update_report(model_data, 'chosen model_data', layer2andmore_chosen_model_data[model_id])
        base_layer_results_repo.update_report(model_data, 'nfolds', oof_nfolds)
        base_layer_results_repo.update_report(model_data, 'stratified', stratified)

    if y_test is not None:
        for k, v in layer2andmore_est_preds.items():
            base_layer_results_repo.update_report(k, 'test_score', metrics_callback(y_test, v))

    base_layer_results_repo.save()

    # if save_report:
    #     stacknet_report = base_layer_results_repo.get_report()
    #     stacknet_report_file = oof_path + 'stacknet_report.csv'
    #     stacknet_report.to_csv(stacknet_report_file, index=False)
    #     module_logger.info('StackNet Report saved at {}'.format(stacknet_report_file))


def _get_metrics_callback(metric):
    metrics_callback = None
    if metric == 'auc':
        metrics_callback = roc_auc_score
    elif 'logloss' in metric:
        metrics_callback = log_loss
    else:
        module_logger.warning('metric is NOT auc, metric callback will be None')
    return metrics_callback
