import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from scipy.sparse import csr_matrix, hstack, vstack
from keras.layers import Dense, Embedding, Input, LSTM, GRU, Bidirectional, \
    GlobalMaxPool1D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import lightgbm as lgb
from automl_libs import nn_libs
import logging
module_logger = logging.getLogger(__name__)
import pdb


class BaseLayerEstimator(ABC):
    @staticmethod
    def _calculate_nb(x, y):
        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        return csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Params:
            x_train: np array
            y_train: pd series
        """
        pass

    @abstractmethod
    def predict(self, x_train):
        pass


class SklearnBLE(BaseLayerEstimator):
    def __init__(self, clf, nb=False, seed=0, params={}, per_label_params={}, need_calibrated_classifier_cv=False):
        """
        Note:
            1. If need to set params for different labels, let params={} when constructing
                so you can set seed, then use set_params() to set params per label
            2. per_label_params: dict. key: label. value: params for the label
            3. For estimators like Linear SVC, CalibratedClassifierCV is needed
        """
        self.clf = clf
        self._nb = nb
        params['random_state'] = seed
        self.per_label_params = per_label_params
        self._seed = seed
        self.params = params
        self._need_calibrated_classifier_cv = need_calibrated_classifier_cv
        self._init_model()

    def _init_model(self):
        if self._need_calibrated_classifier_cv:
            self.model = CalibratedClassifierCV(self.clf(**self.params))
        else:
            self.model = self.clf(**self.params)

    def set_params_for_label(self, label):
        """
        if need to set params for different labels, let params={} when constructing,
        pass in per_label_params, and use this method to set params per label
        """
        self.params = self.per_label_params[label]
        self.params['random_state'] = self._seed
        self._init_model()

    def train(self, x, y):
        if self._nb:
            self._r = self._calculate_nb(x, y.values)
            x = x.multiply(self._r)
        self.model.fit(x, y)

    def cv(self, x, y, nfolds=5, scoring=None):
        """
        Available scoring: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
        if scoring == 'auc':
            scoring = 'roc_auc'
        if scoring is not None:
            score = cross_val_score(self.model, x, y, cv=nfolds, scoring=scoring)
        else:
            score = cross_val_score(self.model, x, y, cv=nfolds)
        return score

    def predict(self, x):
        if self._nb:
            x = x.multiply(self._r)
        return self.model.predict_proba(x)[:, 1]

    def feature_importance(self):
        try:
            return self._clf.feature_importance
        except:  # TODO: give a specific exception
            print('feature_importance not supported for this model')


class LightgbmBLE(BaseLayerEstimator):
    def __init__(self, params={}, nb=False, seed=0):
        self._nb = nb
        self._seed = seed
        self._categorical_feature = params.pop('categorical_feature', 'auto')
        self._num_boost_round = params.pop('best_round', 100)
        self._verbose_eval = params.pop('verbose_eval', 1)
        self._train_params = params
        self._model = None
        self._r = None

    # def set_params(self, params):
    #     """
    #     if need to set params for different labels, let params={} when constructing
    #     so you can set seed, and use this one to set params per label
    #     """
    #     self.params = params
    #     self._train_params['data_random_seed'] = self._seed

    def train(self, x, y, valid_set_percent=0):
        """
        Params:
            x: np/scipy/ 2-d array or matrix
            y: pandas series
            valid_set_percent: (float, 0 to 1).
                    0: no validation set. (imposible to use early stopping)
                    1: use training set as validation set (to check underfitting, and early stopping)
                    >0 and <1: use a portion of training set as validation set. (to check overfitting, and early stopping)

        """
        if self._nb:
            self._r = self._calculate_nb(x, y.values)
            x = x.multiply(self._r)

        if valid_set_percent != 0:
            if valid_set_percent > 1 or valid_set_percent < 0:
                raise ValueError('valid_set_percent must >= 0 and <= 1')
            if valid_set_percent != 1:
                x, x_val, y, y_val = train_test_split(x, y, test_size=valid_set_percent)

        lgb_train = lgb.Dataset(x, y, categorical_feature=self._categorical_feature)
        if valid_set_percent != 0:
            if valid_set_percent == 1:
                print('Evaluating using training set')
                self._model = lgb.train(self._train_params, lgb_train, valid_sets=lgb_train,
                                        num_boost_round=self._num_boost_round, verbose_eval=self._verbose_eval)
            else:
                lgb_val = lgb.Dataset(x_val, y_val, categorical_feature=self._categorical_feature)
                print('Evaluating using validation set ({}% of training set)'.format(valid_set_percent * 100))
                self._model = lgb.train(self._train_params, lgb_train, valid_sets=lgb_val,
                                        num_boost_round=self._num_boost_round, verbose_eval=self._verbose_eval)
        else:
            print('No evaluation set, thus not possible to use early stopping. Please train with your best params.')
            self._model = lgb.train(self._train_params, train_set=lgb_train, valid_sets=lgb_train,
                                    num_boost_round=self._num_boost_round, verbose_eval=self._verbose_eval)

    def predict(self, x):
        if self._nb:
            x = x.multiply(self._r)
        if self._model.best_iteration > 0:
            print('best_iteration {} is chosen.'.format(self._model.best_iteration))
            result = self._model.predict(x, num_iteration=self._model.best_iteration)
        else:
            result = self._model.predict(x)
        return result

    @property
    def model_(self):
        """Get the number of features of fitted model."""
        if self._model is None:
            raise NotFittedError('No model found. Need to call train beforehand.')
        return self._model

    @property
    def r_(self):
        """Get the number of features of fitted model."""
        if self._r is None:
            raise NotFittedError('_r is not calculated. check if nb is set to true')
        return self._r


class NNBLE(BaseLayerEstimator):
    def __init__(self, params={}, seed=0):
        self._seed = seed
        self._categorical_feature = params.pop('categorical_feature')
        self._best_epoch = params.pop('best_epoch')
        self._verbose_eval = params.pop('verbose_eval', 1)
        self._pred_batch_size = params.pop('pred_batch_size')
        self._params = params
        self._model = None
        self._te_dict = None
        self._test_dict = None

    def train(self, x_tr, y_tr, x_te, y_te, x_test):
        """
        Params:
            x_tr: df. train
            y_tr: array-like. Will be converted to np array of shape (N,)
            x_te: df. validation x
            y_te: array-like. validation y. if None, then no validation, and x_te will be ignored
            x_test: df. real test data
        """
        self._model, tr_dict, self._te_dict, self._test_dict = \
            nn_libs.get_model(self._params, x_tr, x_te, x_test, self._categorical_feature)
        val_data = []
        if y_te is not None:
            val_data = [self._te_dict, y_te]
        y_tr = np.array(y_tr).reshape(-1,)
        self._model.fit(tr_dict, y_tr, validation_data=val_data, epochs=self._best_epoch,
                        batch_size=self._params['batch_size'], verbose=self._verbose_eval)

    def predict(self, case):
        """
        Params:
            case:
                x_te: predict x_te of the fold
                x_test: predict x_test
        """
        if case == 'x_te':
            return self._model.predict(self._te_dict, batch_size=self._pred_batch_size, verbose=self._verbose_eval)
        elif case == 'x_test':
            return self._model.predict(self._test_dict, batch_size=self._pred_batch_size, verbose=self._verbose_eval)

    @property
    def model_(self):
        """Get the number of features of fitted model."""
        if self._model is None:
            raise NotFittedError('No model found. Need to call train beforehand.')
        return self._model

    @property
    def test_dict_(self):
        """Get the number of features of fitted model."""
        if self._test_dict is None:
            raise NotFittedError('No test_dict found. Need to call train beforehand.')
        return self._test_dict


class RnnBLE(BaseLayerEstimator):
    """
    This class is still under development.
    """

    def __init__(self, window_length, n_features, label_cols, rnn_units=50, dense_units=50, dropout=0.1,
                 mode='LSTM', bidirection=True):
        self._window_length = window_length
        self._n_features = n_features
        self._label_cols = label_cols
        self._rnn_units = rnn_units
        self._dense_units = dense_units
        self._dropout = dropout
        self._mode = mode
        self._bidirection = bidirection
        self.init_model()

    def init_model(self, load_model=False, load_model_file=None):
        self._model = RnnBLE.get_lstm_model(self._window_length, self._n_features, self._label_cols,
                                            self._rnn_units, self._dense_units, self._dropout,
                                            self._mode, self._bidirection, load_model, load_model_file)

    @staticmethod
    def get_lstm_model(window_length, n_features, label_cols, rnn_units, dense_units,
                       dropout, mode, bidirection, load_model, load_model_file):
        input = Input(shape=(window_length, n_features))
        rnn_layer = LSTM(rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        if mode == 'GRU':
            rnn_layer = GRU(rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        if bidirection:
            x = Bidirectional(rnn_layer)(input)
        else:
            x = rnn_layer(input)
        x = GlobalMaxPool1D()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(len(label_cols), activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x)

        if (load_model):
            print('load model: ' + str(load_model_file))
            model.load_weights(load_model_file)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self, epochs, x_train=None, y_train=None, batch_size=None, callbacks=None,
              validation_split=0.0, validation_data=None, data_gen=None,
              training_steps_per_epoch=None, load_model=False, load_model_file=None):
        if load_model:
            if load_model_file is None:
                raise ValueError('Since load model is True, please provide the load_model_file (path of the model)')
            else:
                self.init_model(load_model, load_model_file)
        if data_gen is None:
            if x_train is None or y_train is None or batch_size is None:
                raise ValueError('Since not training with data generator, please provide: x_train, y_train, batch_size')
            print('training without datagen')
            self._model.fit(x_train, y_train, batch_size=batch_size, validation_split=validation_split, epochs=epochs,
                            callbacks=callbacks)
            return self._model  # for chaining
        else:
            if training_steps_per_epoch is None:
                raise ValueError('training_steps_per_epoch can not be None when using data_gen')
            # with generator:
            print('training with datagen')

            self._model.fit_generator(
                generator=data_gen,
                steps_per_epoch=training_steps_per_epoch,
                epochs=epochs,
                validation_data=validation_data,  # (x_val, y_val)
                callbacks=callbacks
            )
            return self._model

    def predict(self, x, load_model_file=None):
        if load_model_file is not None:
            self._model.load_weights(load_model_file)
        return self._model.predict(x, verbose=1)  # , batch_size=1024)


class OneVSOneRegBLE(BaseLayerEstimator):
    """
    example
        aa = OneVSOneReg(train_tfidf, train[label_cols], model='logistic')
        aa.setModelName('svc')
        aa.train(train_tfidf,train['toxic'], 'toxic')
        aa.predict(test_tfidf, 'toxic')
    """

    def __init__(self, x_train, y_train, model='logistic'):
        """
        x_train: sparse matrix, raw tfidf
        y_train: dataframe, with only label columns. should be 6 columns in total
        model: only support logistic or svc
        """
        self.r = {}
        self.setModelName(model)
        assert self.model_name in ['logistic', 'svc']
        self.param = {}
        self.param['logistic'] = {'identity_hate': 9.0,
                                  'insult': 1.5,
                                  'obscene': 1.0,
                                  'severe_toxic': 4.0,
                                  'threat': 9.0,
                                  'toxic': 2.7}
        self.param['svc'] = {'identity_hate': 0.9,
                             'insult': 0.15,
                             'obscene': 0.15,
                             'severe_toxic': 0.15,
                             'threat': 1.0,
                             'toxic': 0.29}

        for col in y_train.columns:
            # print('calculating naive bayes for {}'.format(col))
            self.r[col] = np.log(self.pr(1, y_train[col].values, x_train) / self.pr(0, y_train[col], x_train))
            # print('initializing done')
            # print('OneVsOne is using {} kernel'.format(self.model_name))

    def setModelName(self, name):
        self.model_name = name
        assert self.model_name in ['logistic', 'svc']
        # print('OneVsOne is using {} kernel'.format(self.model_name))

    def pr(self, y_i, y, train_features):
        p = train_features[np.array(y == y_i)].sum(0)
        return (p + 1) / (np.array(y == y_i).sum() + 1)

    def oneVsOneSplit(self, x_train, y_train, label):
        # print('Starting One vs One dataset splitting')
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        model_train = x_train[np.array(y_train == 1)]
        y_model_train = y_train[np.array(y_train == 1)]
        non_model_train = x_train[np.array(y_train == 0)]
        non_model_train = non_model_train[:model_train.shape[0]]
        y_non_model_train = y_train[np.array(y_train == 0)]
        y_non_model_train = y_non_model_train[:model_train.shape[0]]
        x_model_stack = vstack([model_train, non_model_train])
        y_model_stack = np.concatenate([y_model_train, y_non_model_train])
        x_nb = x_model_stack.multiply(self.r[label]).tocsr()
        y_nb = y_model_stack
        # print('splitting done!')
        return (x_nb, y_nb)

    def train(self, x_train, y_train, label):
        ### construct one vs one
        x_nb, y_nb = self.oneVsOneSplit(x_train, y_train, label)
        ### start training
        if self.model_name is 'logistic':
            # print('start training logistic regression')
            self.model = LogisticRegression(C=self.param['logistic'][label])
            self.model.fit(x_nb, y_nb)
            # print('training done')

        else:
            # print('start training linear svc regression')
            lsvc = LinearSVC(C=self.param['svc'][label])
            self.model = CalibratedClassifierCV(lsvc)
            self.model.fit(x_nb, y_nb)
            # print('training done')

    def predict(self, x_test, label):
        # print('applying naive bayes to dataset')
        x_nb_test = x_test.multiply(self.r[label]).tocsr()
        # print('predicting')
        pred = self.model.predict_proba(x_nb_test)[:, 1]
        # print('predicting done')
        return pred

