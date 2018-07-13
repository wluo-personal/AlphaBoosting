import time
import os
import gc
from datetime import timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from automl_libs import utils, nn_libs
import pdb
import logging
module_logger = logging.getLogger(__name__)


def gs(X_train, y_train, X_val, y_val, categorical_feature, search_rounds,
       gs_record_dir, gs_params_gen, gs_models, cv, nfold,
       verbose_eval, do_preds, X_test, y_test, preds_save_path, suppress_warning, **kwargs):

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
                if gs_model == 'logreg' or gs_model == 'svc':
                    params, run_id = _svc_logreg_gs(X_train, y_train, X_val, y_val, categorical_feature,
                                     gs_params_gen, gs_model, cv, nfold, verbose_eval,
                                     do_preds, X_test, y_test, preds_save_path)
                elif gs_model == 'lgb':
                    params, run_id = _lgb_gs(X_train, y_train, X_val, y_val, categorical_feature,
                                     gs_params_gen, gs_model, cv, nfold, verbose_eval,
                                     do_preds, X_test, y_test, preds_save_path)
                elif gs_model == 'nn':
                    params, run_id = _nn_gs(X_train, y_train, X_val, y_val, categorical_feature,
                                    gs_params_gen, gs_model, verbose_eval,
                                    do_preds, X_test, y_test, preds_save_path)

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
                    module_logger.info(filename_for_gs_results + ' updated')

            except Exception as e:
                if 'ResourceExhaustedError' in str(type(e)): # can't catch this error directly...
                    module_logger.warning('Oops! ResourceExhaustedError. Continue next round')
                    continue
                else:
                    raise  # throw the exception


def _svc_logreg_gs(X_train, y_train, X_val, y_val, categorical_feature,
            gs_params_gen, gs_model, cv, nfold, verbose_eval,
            do_preds, X_test, y_test, preds_save_path):
    """
    TODO: untested. bad performance, hence not priority.
    """
    params, seed = gs_params_gen(gs_model)
    metric = params['metric']
    md = None
    if gs_model == 'svc':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        md = CalibratedClassifierCV(LinearSVC(**params))
        # n_estimators = 8
        # max_samples = 0.2
        # md = BaggingClassifier(lsvc, max_samples=max_samples,
        #                        n_estimators=n_estimators, n_jobs=n_estimators)
    elif gs_model == 'logreg':
        md = LogisticRegression()#penalty=params['penalty'], dual=params['dual'], C=params['C'], n_jobs=8)
    run_id = utils.get_random_string()  # also works as the index of the result dataframe

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(lgb_params)
    params['timestamp'] = utils.get_time()
    gs_start_time = time.time()
    if cv:
        # use ALL data to do cv
        total_X_train = pd.concat([X_train, X_val])
        total_y_train = pd.concat([y_train, y_val])
        scores = cross_val_score(md, total_X_train, total_y_train, cv=nfold, scoring=metric)
        cv_val_metric = np.mean(scores)
        params['val_' + metric] = cv_val_metric
        module_logger.info('val_{}: {:.5f} (cv, no train_{})'.format(metric, cv_val_metric, metric))
        params['cv'] = True
    else:
        md.fit(X_train, y_train)
        train_pred = md.predict_proba(X_train)[:, 1]
        train_metric = roc_auc_score(y_train, train_pred)
        val_pred = md.predict_proba(X_val)[:, 1]
        val_metric = roc_auc_score(y_val, val_pred)
        params['val_' + metric] = val_metric
        params['train_' + metric] = train_metric
        module_logger.info('val_{}: {:.5f} | train_{}: {:.5f} (not cv)'.format(metric, val_metric, metric, train_metric))

    # time spent in this round of search, in format hh:mm:ss
    gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
    params['gs_timespent'] = gs_elapsed_time_as_hhmmss

    if do_preds:
        predict_start_time = time.time()
        module_logger.info('[do_preds] is True, generating predictions ...')
        module_logger.info('Retrain model using all data...')
        md.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        train_pred = md.predict_proba(X_train)[:, 1]
        module_logger.info('Training done. Train_{}: {:.5f} | {} features'
                           .format(metric, roc_auc_score(y_train, train_pred), X_train.shape[1]))
        y_test_pred = md.predict_proba(X_test)[:, 1]

        if y_test is not None:
            module_logger.info('(_{}_gs) roc of test: {}'.format(gs_model, roc_auc_score(y_test, y_test_pred)))

        np.save(preds_save_path + gs_model + '_preds_{}'.format(run_id), y_test_pred)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        params['pred_timespent'] = predict_elapsed_time_as_hhmmss
        module_logger.info('{} predictions({}) saved in {}.'.format(gs_model, run_id, preds_save_path))

    return params, run_id


def _lgb_gs(X_train, y_train, X_val, y_val, categorical_feature,
            gs_params_gen, gs_model, cv, nfold, verbose_eval,
            do_preds, X_test, y_test, preds_save_path):
    lgb_params, seed = gs_params_gen(gs_model)
    metric = lgb_params['metric']
    run_id = utils.get_random_string()  # also works as the index of the result dataframe

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(lgb_params)
    lgb_params['timestamp'] = utils.get_time()
    gs_start_time = time.time()
    if cv:
        # use ALL data to do cv
        lgb_train = lgb.Dataset(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]),
                                categorical_feature=categorical_feature)
        eval_hist = lgb.cv(lgb_params, lgb_train, nfold=nfold,
                           categorical_feature=categorical_feature, verbose_eval=verbose_eval, seed=seed)
        del lgb_train; gc.collect()
        best_round = len(eval_hist[metric + '-mean'])
        lgb_params['best_round'] = best_round
        cv_val_metric = eval_hist[metric + '-mean'][-1]
        lgb_params['val_' + metric] = cv_val_metric
        module_logger.info('val_{}: {:.5f} (cv, no train_{})'.format(metric, cv_val_metric, metric))
        lgb_params['cv'] = True
    else:
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
        lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=categorical_feature)
        model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_val], verbose_eval=verbose_eval)
        del lgb_train, lgb_val; gc.collect()
        lgb_params['best_round'] = model.best_iteration
        val_metric = model.best_score['valid_1'][metric]
        train_metric = model.best_score['training'][metric]
        lgb_params['val_' + metric] = val_metric
        lgb_params['train_' + metric] = train_metric
        module_logger.info('val_{}: {:.5f} | train_{}: {:.5f} (not cv)'.format(metric, val_metric, metric, train_metric))

    # time spent in this round of search, in format hh:mm:ss
    gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
    lgb_params['gs_timespent'] = gs_elapsed_time_as_hhmmss

    if do_preds:
        predict_start_time = time.time()
        module_logger.info('[do_preds] is True, generating predictions ...')
        module_logger.info('Retrain model using best_round and all data...')
        best_round = lgb_params['best_round']
        lgb_all_data = lgb.Dataset(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]),
                                   categorical_feature=categorical_feature)
        model = lgb.train(lgb_params, lgb_all_data, valid_sets=lgb_all_data,
                          num_boost_round=best_round, verbose_eval=int(0.2 * best_round))
        module_logger.info('Training done. Iteration: {} | train_{}: {:.5f} | {} features'
                           .format(model.current_iteration(), metric,
                                   model.best_score['training'][metric], model.num_feature()))
        del lgb_all_data; gc.collect()
        y_test_pred = model.predict(X_test)

        if y_test is not None:
            module_logger.info('(_nn_gs) roc of test: {}'.format(roc_auc_score(y_test, y_test_pred)))

        np.save(preds_save_path + 'lgb_preds_{}'.format(run_id), y_test_pred)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        lgb_params['pred_timespent'] = predict_elapsed_time_as_hhmmss
        module_logger.info('LGB predictions({}) saved in {}.'.format(run_id, preds_save_path))

    # remove params not needed to be recorded in grid search history csv
    lgb_params.pop('categorical_column', None)
    lgb_params.pop('verbose', None)

    return lgb_params, run_id


def _nn_gs(X_train, y_train, X_val, y_val, categorical_feature,
           gs_params_gen, gs_model, verbose_eval,
           do_preds, X_test, y_test, preds_save_path):
    nn_params, seed = gs_params_gen(gs_model)
    # time.sleep(1)  # sleep 1 sec to make sure the run_id is unique
    run_id = utils.get_random_string()

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
    cb = []
    if nn_params['monitor'] == 'val_auc':
        # include the following before EarlyStopping
        cb.append(nn_libs.RocAucMetricCallback(validation_data=(valid_dict, y_val),
                                               predict_batch_size=pred_batch_size))
    cb.extend([
        EarlyStopping(monitor=nn_params['monitor'], mode=nn_params['mode'], patience=nn_params['patience'], verbose=2),
        nn_libs.LearningRateTracker(include_on_batch=False),
        ModelCheckpoint(saved_model_file_name, monitor=nn_params['monitor'], verbose=1, save_best_only=True, mode='max')
        # LearningRateScheduler(lambda x: lr * (lr_decay ** x))
    ])
    model.fit(train_dict, y_train, validation_data=[valid_dict, y_val],
              epochs=nn_params['max_ep'], batch_size=nn_params['batch_size'], verbose=verbose_eval, callbacks=cb)

    hist = model.history
    bst_epoch = np.argmax(hist.history[nn_params['monitor']])
    if nn_params['monitor'] == 'val_auc':
        val_auc = hist.history['val_auc'][bst_epoch]
        nn_params['val_auc'] = val_auc  # hence the val_auc score will be later saved in csv
        module_logger.info('val_auc: {:.5f}'.format(val_auc))
    trn_loss = hist.history['loss'][bst_epoch]
    trn_acc = hist.history['acc'][bst_epoch]  # TODO: acc might not be there if regression problem
    val_loss = hist.history['val_loss'][bst_epoch]
    val_acc = hist.history['val_acc'][bst_epoch]
    nn_params['best_epoch'] = bst_epoch + 1
    nn_params['trn_loss'] = trn_loss
    nn_params['trn_acc'] = trn_acc
    nn_params['val_loss'] = val_loss
    nn_params['val_acc'] = val_acc

    module_logger.info('val_loss: {:.5f} | train_loss: {:.5f} (not cv)'.format(val_loss, trn_loss))

    # time spent in this round of search, in format hh:mm:ss
    gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
    nn_params['gs_timespent'] = gs_elapsed_time_as_hhmmss

    if do_preds:
        predict_start_time = time.time()
        module_logger.info('[do_preds] is True, generating predictions ...')
        model.load_weights(saved_model_file_name)
        y_test_pred = model.predict(test_dict, batch_size=pred_batch_size, verbose=verbose_eval)

        if y_test is not None:
            module_logger.info('(_nn_gs) roc of test: {}'.format(roc_auc_score(y_test, y_test_pred)))

        np.save(preds_save_path + 'nn_preds_{}'.format(run_id), y_test_pred)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        nn_params['pred_timespent'] = predict_elapsed_time_as_hhmmss
        module_logger.info('NN predictions({}) saved in {}.'.format(run_id, preds_save_path))

    return nn_params, run_id
