from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import gc
import logging
import time
from datetime import timedelta
from sklearn.metrics import roc_auc_score
from automl_libs import utils, nn_libs
import lightgbm as lgb
import xgboost as xgb
import catboost as catb
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import pdb


def get_gs_instance(X_train, y_train, X_val, y_val, categorical_feature, gs_params_gen, gs_model, verbose_eval, X_test):
    if gs_model == 'lgb':
        return LgbGS(X_train, y_train, X_val, y_val, categorical_feature, gs_params_gen, gs_model, verbose_eval, X_test)
    elif gs_model == 'xgb':
        return XgbGS(X_train, y_train, X_val, y_val, categorical_feature, gs_params_gen, gs_model, verbose_eval, X_test)
    elif gs_model == 'catb':
        return CatbGS(X_train, y_train, X_val, y_val, categorical_feature, gs_params_gen, gs_model, verbose_eval, X_test)
    elif gs_model == 'nn':
        return NNGS(X_train, y_train, X_val, y_val, categorical_feature, gs_params_gen, gs_model, verbose_eval, X_test)
    else:
        raise ValueError('{} is unknown. Available options: lgb, xgb, catb, nn'.format(gs_model))


class GridSearch(ABC):
    def __init__(self, X_train, y_train, X_val, y_val, categorical_feature,
                 gs_params_gen, gs_model, verbose_eval, X_test):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.categorical_feature = categorical_feature
        self.gs_params_gen = gs_params_gen
        self.gs_model = gs_model
        self.verbose_eval = verbose_eval
        self.X_test = X_test
        self.gs_params, self.seed = self.gs_params_gen(self.gs_model)

        self.model = None
        self.metric = None
        self.best_round = None
        self.catb_categorical_features_indices = None
        self.xgb_nbr = None
        self.xgb_esr = None
        self.nn_train_dict = None
        self.nn_valid_dict = None
        self.nn_test_dict = None
        self.nn_saved_model_file_name = None

        self.init_more()

    def init_more(self):
        pass

    @abstractmethod
    def cv(self, nfold, stratified):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def pred(self):
        pass


class LgbGS(GridSearch):

    def init_more(self):
        self.metric = self.gs_params['metric']

    def cv(self, nfold, stratified):
        gs_start_time = time.time()
        pdb.set_trace()
        lgb_train = lgb.Dataset(pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val]),
                                categorical_feature=self.categorical_feature)
        eval_hist = lgb.cv(self.gs_params, lgb_train, nfold=nfold, stratified=stratified,
                           categorical_feature=self.categorical_feature, verbose_eval=self.verbose_eval, seed=self.seed)
        del lgb_train; gc.collect()
        best_round = len(eval_hist[self.metric + '-mean'])
        self.best_round = best_round
        cv_val_metric = eval_hist[self.metric + '-mean'][-1]
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return best_round, cv_val_metric, -1, gs_elapsed_time_as_hhmmss

    def val(self):
        gs_start_time = time.time()
        lgb_train = lgb.Dataset(self.X_train, self.y_train, categorical_feature=self.categorical_feature)
        lgb_val = lgb.Dataset(self.X_val, self.y_val, categorical_feature=self.categorical_feature)
        model = lgb.train(self.gs_params, lgb_train, valid_sets=[lgb_train, lgb_val], verbose_eval=self.verbose_eval)
        del lgb_train, lgb_val; gc.collect()
        best_round = model.best_iteration
        self.best_round = best_round
        val_metric = model.best_score['valid_1'][self.metric]
        train_metric = model.best_score['training'][self.metric]
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return best_round, val_metric, train_metric, gs_elapsed_time_as_hhmmss

    def pred(self):
        predict_start_time = time.time()
        pdb.set_trace()
        self.logger.info('Retrain model using best_round [{}] and all data...'.format(self.best_round))
        lgb_all_data = lgb.Dataset(pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val]),
                                   categorical_feature=self.categorical_feature)
        model = lgb.train(self.gs_params, lgb_all_data, valid_sets=lgb_all_data,
                          num_boost_round=self.best_round, verbose_eval=int(0.2 * self.best_round))
        train_alldata_metric = model.best_score['training'][self.metric]
        self.logger.info('Training done. Iteration: {} | train_{}: {:.5f} | {} features'
                           .format(model.current_iteration(), self.metric,
                                   train_alldata_metric, model.num_feature()))
        # self.gs_res_dict['train_all_'+self.metric] = train_alldata_metric
        del lgb_all_data; gc.collect()
        y_test_pred = model.predict(self.X_test)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        # self.gs_res_dict['pred_timespent'] = predict_elapsed_time_as_hhmmss
        # module_logger.info('LGB predictions({}) saved in {}.'.format(run_id, preds_save_path))
        return y_test_pred, train_alldata_metric, predict_elapsed_time_as_hhmmss


class CatbGS(GridSearch):

    def init_more(self):
        self.model = catb.CatBoostClassifier(**self.gs_params)
        self.catb_categorical_features_indices = [self.X_train.columns.tolist().index(col)
                                                  for col in self.categorical_feature]
        self.metric = self.gs_params['eval_metric']

    def cv(self, nfold, stratified):
        gs_start_time = time.time()
        # use ALL data to do cv
        eval_hist = catb.cv(
            catb.Pool(pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val]),
                      cat_features=self.catb_categorical_features_indices),
            self.model.get_params(),
            nfold=nfold,
            stratified=stratified,
            verbose_eval=self.verbose_eval
        )
        best_round = eval_hist['test-' + self.metric + '-mean'].values.argmax()
        self.gs_params['iterations'] = best_round
        cv_val_metric = eval_hist['test-' + self.metric + '-mean'][best_round]
        cv_train_metric = eval_hist['train-' + self.metric + '-mean'][best_round]
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return best_round, cv_val_metric, cv_train_metric, gs_elapsed_time_as_hhmmss

    def val(self):
        gs_start_time = time.time()
        self.model.fit(self.X_train, self.y_train, cat_features=self.catb_categorical_features_indices,
                       eval_set=[(self.X_val, self.y_val), (self.X_train, self.y_train)],
                       verbose_eval=self.verbose_eval)
        best_round = self.model.tree_count_
        self.gs_params['iterations'] = best_round
        val_metric = roc_auc_score(self.y_val, self.model.get_test_evals()[0][0])
        train_metric = roc_auc_score(self.y_train, self.model.get_test_evals()[1][0])
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return best_round, val_metric, train_metric, gs_elapsed_time_as_hhmmss

    def pred(self):
        predict_start_time = time.time()
        best_round = self.gs_params['iterations']
        self.logger.info('Retrain model using best_round [{}] and all data...'.format(best_round))
        model = catb.CatBoostClassifier(**self.gs_params)
        X_all = pd.concat([self.X_train, self.X_val])
        y_all = pd.concat([self.y_train, self.y_val])
        model.fit(X_all, y_all, cat_features=self.catb_categorical_features_indices, eval_set=(X_all, y_all),
                  verbose_eval=int(0.2 * self.gs_params['iterations']))
        train_alldata_metric = roc_auc_score(y_all, model.get_test_evals()[0][0])
        self.logger.info('Training done. Iteration: {} | train_{}: {:.5f}'
                         .format(best_round, self.metric, train_alldata_metric))
        del X_all, y_all; gc.collect()
        y_test_pred = model.predict_proba(self.X_test)[:, 1]
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        return y_test_pred, train_alldata_metric, predict_elapsed_time_as_hhmmss


class XgbGS(GridSearch):

    def init_more(self):
        self.xgb_nbr = self.gs_params['num_boost_round']
        self.xgb_esr = self.gs_params['early_stopping_rounds']
        self.metric = self.gs_params['eval_metric']

    def cv(self, nfold, stratified):
        gs_start_time = time.time()
        # use ALL data to do cv
        xgb_train = xgb.DMatrix(pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val]))
        eval_hist = xgb.cv(self.gs_params, xgb_train, nfold=nfold, stratified=stratified,
                           early_stopping_rounds=self.xgb_esr, num_boost_round=self.xgb_nbr,
                           verbose_eval=self.verbose_eval, seed=self.seed)
        del xgb_train;
        gc.collect()
        self.best_round = len(eval_hist['test-' + self.metric + '-mean'])
        cv_val_metric = eval_hist['test-' + self.metric + '-mean'].values[-1]
        cv_train_metric = eval_hist['train-' + self.metric + '-mean'].values[-1]
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return self.best_round, cv_val_metric, cv_train_metric, gs_elapsed_time_as_hhmmss

    def val(self):
        gs_start_time = time.time()
        xgb_train = xgb.DMatrix(self.X_train, self.y_train)
        xgb_val = xgb.DMatrix(self.X_val, self.y_val)
        evals_res = {}
        model = xgb.train(self.gs_params, xgb_train, num_boost_round=self.xgb_nbr, early_stopping_rounds=self.xgb_esr,
                          evals=[(xgb_val, 'val')], evals_result=evals_res, verbose_eval=self.verbose_eval)
        self.best_round = model.best_iteration
        del xgb_train, xgb_val;
        gc.collect()
        val_metric = evals_res['val'][self.metric][-1]
        # train_metric = evals_res['train'][metric][-1]
        # xgb_params['train_' + metric] = train_metric
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        return self.best_round, val_metric, -1, gs_elapsed_time_as_hhmmss

    def pred(self):
        predict_start_time = time.time()
        best_round = self.best_round
        self.logger.info('Retrain model using best_round [{}] and all data...'.format(best_round))
        xgb_all_data = xgb.DMatrix(pd.concat([self.X_train, self.X_val]), pd.concat([self.y_train, self.y_val]))
        xgb_test = xgb.DMatrix(self.X_test)
        evals_res = {}
        eval_data_name = 'all_data'
        model = xgb.train(self.gs_params, xgb_all_data, evals=[(xgb_all_data, eval_data_name)], evals_result=evals_res,
                          num_boost_round=best_round, verbose_eval=int(0.2 * best_round))
        train_alldata_metric = evals_res[eval_data_name][self.metric][-1]
        self.logger.info('Training done. Iteration: {} | train_{}: {:.5f}'
                           .format(best_round, self.metric, train_alldata_metric))
        del xgb_all_data;
        gc.collect()
        y_test_pred = model.predict(xgb_test)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        return y_test_pred, train_alldata_metric, predict_elapsed_time_as_hhmmss


class NNGS(GridSearch):

    def init_more(self):
        if self.gs_params['monitor'] == 'val_auc':
            self.metric = 'auc'
        else:
            self.metric = 'notdefined'
        self.model, self.nn_train_dict, self.nn_valid_dict, self.nn_test_dict = \
            nn_libs.get_model(self.gs_params, self.X_train, self.X_val, self.X_test, self.categorical_feature)
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')

    def cv(self, nfold, stratified):
        # not applied
        pass

    def val(self):
        # not applied
        pass

    def nn_val(self, saved_model_file_name):
        self.nn_saved_model_file_name = saved_model_file_name
        gs_start_time = time.time()
        cb = []
        if self.gs_params['monitor'] == 'val_auc':
            # include the following before EarlyStopping
            cb.append(nn_libs.RocAucMetricCallback(validation_data=(self.nn_valid_dict, self.y_val),
                                                   predict_batch_size=self.gs_params['pred_batch_size']))
        cb.extend([
            EarlyStopping(monitor=self.gs_params['monitor'], mode=self.gs_params['mode'], patience=self.gs_params['patience'],
                          verbose=2),
            nn_libs.LearningRateTracker(include_on_batch=False),
            ModelCheckpoint(saved_model_file_name, monitor=self.gs_params['monitor'], verbose=1, save_best_only=True,
                            mode='max')
            # LearningRateScheduler(lambda x: lr * (lr_decay ** x))
        ])
        self.model.fit(self.nn_train_dict, self.y_train, validation_data=[self.nn_valid_dict, self.y_val],
                       epochs=self.gs_params['max_ep'], batch_size=self.gs_params['batch_size'],
                       verbose=self.verbose_eval, callbacks=cb)

        hist = self.model.history
        bst_epoch = np.argmax(hist.history[self.gs_params['monitor']])
        if self.gs_params['monitor'] == 'val_auc':
            val_auc = hist.history['val_auc'][bst_epoch]
            self.gs_params['val_auc'] = val_auc  # hence the val_auc score will be later saved in csv
            self.logger.info('val_auc: {:.5f}'.format(val_auc))
        trn_loss = hist.history['loss'][bst_epoch]
        trn_acc = hist.history['acc'][bst_epoch]  # TODO: acc might not be there if regression problem
        val_loss = hist.history['val_loss'][bst_epoch]
        val_acc = hist.history['val_acc'][bst_epoch]
        self.gs_params['best_epoch'] = bst_epoch + 1
        self.gs_params['trn_loss'] = trn_loss
        self.gs_params['trn_acc'] = trn_acc
        self.gs_params['val_loss'] = val_loss
        self.gs_params['val_acc'] = val_acc
        self.logger.info('val_loss: {:.5f} | train_loss: {:.5f} (not cv)'.format(val_loss, trn_loss))
        gs_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - gs_start_time)))
        self.gs_params['gs_timespent'] = gs_elapsed_time_as_hhmmss
        return self.gs_params

    def pred(self):
        predict_start_time = time.time()
        self.model.load_weights(self.nn_saved_model_file_name)
        y_test_pred = self.model.predict(self.nn_test_dict, batch_size=self.gs_params['pred_batch_size'],
                                         verbose=self.verbose_eval)
        predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
        return y_test_pred, -1, predict_elapsed_time_as_hhmmss
