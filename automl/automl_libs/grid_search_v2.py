import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from automl_libs import utils, nn_libs
from automl_libs import grid_search_factory as gsf
import pdb
import logging
module_logger = logging.getLogger(__name__)


def gs(data_name, X_train, y_train, X_val, y_val, categorical_feature, search_rounds,
       gs_record_dir, gs_params_gen, gs_models, cv, nfold, stratified,
       verbose_eval, do_preds, X_test, y_test, auto_sub_func, preds_save_path, suppress_warning, **kwargs):

    if suppress_warning:
        import warnings
        warnings.filterwarnings("ignore")

    if do_preds:
        if not os.path.exists(preds_save_path):
            os.makedirs(preds_save_path)

    for gs_model in gs_models.split('|'):
        for i in range(search_rounds):
            module_logger.info('Grid search {}. round {} of {}'.format(gs_model, i+1, search_rounds))
            try:
                run_id = utils.get_random_string()  # also works as the index of the result dataframe
                gs_instance = gsf.get_gs_instance(X_train, y_train, X_val, y_val, categorical_feature,
                                                  gs_params_gen, gs_model, verbose_eval, X_test)
                metric = gs_instance.metric
                gs_res_dict = gs_instance.gs_params.copy()
                gs_res_dict['timestamp'] = utils.get_time()
                if gs_model == 'nn':
                    nn_saved_models_path = preds_save_path + 'nn_saved_models/'
                    if not os.path.exists(nn_saved_models_path):
                        os.makedirs(nn_saved_models_path)
                    saved_model_file_name = nn_saved_models_path + 'nn_{}.hdf5'.format(run_id)
                    gs_res_dict.update(gs_instance.nn_val(saved_model_file_name))
                else:
                    if cv:
                        best_round, val_metric, train_metric, gs_elapsed_time = gs_instance.cv(nfold, stratified)
                        gs_res_dict['cv'] = True
                        gs_res_dict['cv_nfold'] = nfold
                        gs_res_dict['cv_stratified'] = stratified
                    else:
                        best_round, val_metric, train_metric, gs_elapsed_time = gs_instance.val()
                        gs_res_dict['cv'] = False
                    gs_res_dict['best_round'] = best_round
                    gs_res_dict['val_'+metric] = val_metric
                    gs_res_dict['train_'+metric] = train_metric
                    gs_res_dict['gs_timespent'] = gs_elapsed_time
                    module_logger.info('{} | {}: val_{}: {:.5f} | train_{} {:.5f})'
                                       .format('CV' if cv else 'NOT CV',
                                               gs_model, metric, val_metric, metric, train_metric))

                module_logger.info('GS time spent: {}'.format(gs_res_dict['gs_timespent']))

                if do_preds:
                    module_logger.info('[do_preds] is True, generating predictions ...')
                    y_test_pred, train_alldata_metric, pred_elapsed_time = gs_instance.pred()
                    gs_res_dict['train_all_'+metric] = train_alldata_metric
                    gs_res_dict['pred_timespent'] = pred_elapsed_time
                    module_logger.info('Pred time spent: {}'.format(pred_elapsed_time))

                    pred_npy_file = preds_save_path + '{}_preds_{}'.format(gs_model, run_id)
                    np.save(pred_npy_file, y_test_pred)
                    module_logger.info('{} predictions({}) saved in {}.'.format(gs_model, run_id, preds_save_path))

                    if y_test is not None:
                        module_logger.info(
                            '(_{}_gs) roc of test: {}'.format(gs_model, roc_auc_score(y_test, y_test_pred)))

                    if auto_sub_func is not None:
                        try:
                            preds = np.load(pred_npy_file+'.npy')
                            subfilename = pred_npy_file.split('/')[-1]
                            auto_sub_func(preds, subfilename)
                        except:
                            print('Auto Submission Failed: ', sys.exc_info()[0])

                gs_res_dict.pop('early_stopping_round',None)
                gs_res_dict.pop('early_stopping_rounds',None)
                utils.save_params_and_result(run_id, gs_model, data_name, 'grid_search', gs_res_dict,
                                             ['categorical_column','verbose'], gs_record_dir)

            except Exception as e:
                if 'ResourceExhaustedError' in str(type(e)): # can't catch this error directly...
                    module_logger.warning('Oops! ResourceExhaustedError. Continue next round')
                    continue
                else:
                    raise  # throw the exception


