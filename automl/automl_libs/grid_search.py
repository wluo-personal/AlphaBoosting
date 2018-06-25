import time
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from automl_libs import utils, nn_libs
import logging
module_logger = logging.getLogger(__name__)


def gs(X_train, y_train, X_val, y_val, categorical_feature, search_rounds,
       gs_record_dir, gs_params_gen, gs_model, cv, nfold,
       verbose_eval, do_preds, X_test, preds_save_path, suppress_warning):

    if suppress_warning:
        import warnings
        warnings.filterwarnings("ignore")

    if do_preds:
        if not os.path.exists(preds_save_path):
            os.makedirs(preds_save_path)

    for i in range(search_rounds):
        try:
            if gs_model == 'lgb':
                params, run_id = _lgb_gs(X_train, y_train, X_val, y_val, categorical_feature,
                                 gs_params_gen, gs_model, cv, nfold, verbose_eval,
                                 do_preds, X_test, preds_save_path)
            elif gs_model == 'nn':
                params, run_id = _nn_gs(X_train, y_train, X_val, y_val, categorical_feature,
                                gs_params_gen, gs_model, verbose_eval,
                                do_preds, X_test, preds_save_path)

            # so that [1,2,3] can be converted to "[1,2,3]" and be treated as a whole in csv
            for k, v in params.items():
                if isinstance(v, list):
                    params[k] = '"'+str(v)+'"'
                    #module_logger.debug(params[k])

            res = pd.DataFrame(params, index=[run_id])
            filename_for_gs_results = gs_record_dir + '{}_grid_search.csv'.format(gs_model)
            if not os.path.exists(filename_for_gs_results):
                res.to_csv(filename_for_gs_results)
                module_logger.debug(filename_for_gs_results + ' created')
            else:
                old_res = pd.read_csv(filename_for_gs_results, index_col='Unnamed: 0')
                res = pd.concat([old_res, res])
                res.to_csv(filename_for_gs_results)
                module_logger.debug(filename_for_gs_results + ' updated')

        except Exception as e:
            if 'ResourceExhaustedError' in str(type(e)): # can't catch this error directly... 
                module_logger.warning('Oops! ResourceExhaustedError. Continue next round')
                continue
            else:
                raise  # throw the exception


def _lgb_gs(X_train, y_train, X_val, y_val, categorical_feature,
            gs_params_gen, gs_model, cv, nfold, verbose_eval,
            do_preds, X_test, preds_save_path):
    lgb_params, seed = gs_params_gen(gs_model)
    metric = lgb_params['metric']
    time.sleep(1)  # sleep 1 sec to make sure the run_id is unique
    run_id = int(time.time())  # also works as the index of the result dataframe

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)

    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=categorical_feature)

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(lgb_params)
    lgb_params['timestamp'] = utils.get_time()
    gs_start_time = time.time()
    if cv:
        eval_hist = lgb.cv(lgb_params, lgb_train, nfold=nfold,
                           categorical_feature=categorical_feature, verbose_eval=verbose_eval, seed=seed)
        best_rounds = len(eval_hist[metric + '-mean'])
        lgb_params['best_round'] = best_rounds
        lgb_params['val_' + metric] = eval_hist[metric + '-mean'][-1]
        lgb_params['cv'] = True
    else:
        model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_val], verbose_eval=verbose_eval)
        lgb_params['best_round'] = model.best_iteration
        lgb_params['val_' + metric] = model.best_score['valid_1'][metric]
        lgb_params['train_' + metric] = model.best_score['training'][metric]

    # time spent in this round of search, in format hh:mm:ss
    gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
    lgb_params['gs_timespent'] = gs_elapsed_time_as_hhmmss

    if do_preds:
        predict_start_time = time.time()
        module_logger.debug('[do_preds] is True, generating predictions ...')
        if cv:
            module_logger.debug('Retrain model using cv params and all data...')
            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
            model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_train,
                              num_boost_round=best_rounds, verbose_eval=int(0.2 * best_rounds))
        y_test = model.predict(X_test)
        np.save(preds_save_path + 'lgb_preds_{}'.format(run_id), y_test)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        lgb_params['pred_timespent'] = predict_elapsed_time_as_hhmmss
        module_logger.info('LGB predictions({}) saved in {}.'.format(run_id, preds_save_path))

    # remove params not needed to be recorded in grid search history csv
    lgb_params.pop('categorical_column', None)
    lgb_params.pop('verbose', None)

    return lgb_params, run_id


def _nn_gs(X_train, y_train, X_val, y_val, categorical_feature,
           gs_params_gen, gs_model, verbose_eval,
           do_preds, X_test, preds_save_path):
    nn_params, seed = gs_params_gen(gs_model)
    time.sleep(1)  # sleep 1 sec to make sure the run_id is unique
    run_id = int(time.time())

    nn_params['timestamp'] = utils.get_time()
    gs_start_time = time.time()
    model, train_dict, valid_dict, test_dict = \
        nn_libs.get_model(nn_params, X_train, X_val, X_test, categorical_feature)

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')

    nn_saved_models_path = preds_save_path + 'nn_saved_models/'
    if not os.path.exists(nn_saved_models_path):
        os.makedirs(nn_saved_models_path)
    saved_model_file_name = nn_saved_models_path + 'nn_{}.hdf5'.format(run_id)

    pred_batch_size = nn_params['pred_batch_size']
    cb = [
        nn_libs.RocAucMetricCallback(validation_data=(valid_dict, y_val), predict_batch_size=pred_batch_size),  # include it before EarlyStopping!
        EarlyStopping(monitor='roc_auc_val', mode='max', patience=nn_params['patience'], verbose=2),
        nn_libs.LearningRateTracker(include_on_batch=False),
        ModelCheckpoint(saved_model_file_name, monitor='roc_auc_val', verbose=1, save_best_only=True, mode='max')
        # LearningRateScheduler(lambda x: lr * (lr_decay ** x))
    ]
    model.fit(train_dict, y_train, validation_data=[valid_dict, y_val],
              epochs=nn_params['max_ep'], batch_size=nn_params['batch_size'], verbose=verbose_eval, callbacks=cb)

    hist = model.history
    bst_epoch = np.argmax(hist.history['roc_auc_val'])
    trn_loss = hist.history['loss'][bst_epoch]
    trn_acc = hist.history['acc'][bst_epoch]
    val_loss = hist.history['val_loss'][bst_epoch]
    val_acc = hist.history['val_acc'][bst_epoch]
    val_auc = hist.history['roc_auc_val'][bst_epoch]
    nn_params['trn_loss'] = trn_loss
    nn_params['trn_acc'] = trn_acc
    nn_params['val_loss'] = val_loss
    nn_params['val_acc'] = val_acc
    nn_params['val_auc'] = val_auc

    # time spent in this round of search, in format hh:mm:ss
    gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
    nn_params['gs_timespent'] = gs_elapsed_time_as_hhmmss

    if do_preds:
        predict_start_time = time.time()
        module_logger.debug('[do_preds] is True, generating predictions ...')
        model.load_weights(saved_model_file_name)
        y_test = model.predict(test_dict, batch_size=pred_batch_size, verbose=verbose_eval)
        np.save(preds_save_path + 'nn_preds_{}'.format(run_id), y_test)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        nn_params['pred_timespent'] = predict_elapsed_time_as_hhmmss
        module_logger.info('NN predictions({}) saved in {}.'.format(run_id, preds_save_path))

    nn_params.pop('pred_batch_size')
    return nn_params, run_id
