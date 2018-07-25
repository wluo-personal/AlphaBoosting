import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from scipy.sparse import csr_matrix, hstack, vstack
from enum import Enum
import pickle
import copy
import sys
import logging
from automl_libs import SklearnBLE, NNBLE
module_logger = logging.getLogger(__name__)
import pdb


class ModelName(Enum):
    XGB = 1
    NBXGB = 2
    XGB_PERLABEL = 3
    NBXGB_PERLABEL = 4
    LGB = 5
    NBLGB = 6
    LGB_PERLABEL = 7
    NBLGB_PERLABEL = 8
    LOGREG = 9
    NBLOGREG = 10 # NBSVM
    LOGREG_PERLABEL = 11
    NBLOGREG_PERLABEL = 12
    LSVC = 13
    NBLSVC = 14
    LSVC_PERLABEL = 15
    NBLSVC_PERLABEL = 16
    DTC = 17 # DecisionTreeClassifier
    RF = 18 # random forest
    ET = 19 # extra trees
    NN = 20
    ONESVC = 21
    ONELOGREG = 22
    CATB = 23


class BaseLayerDataRepo:
    def __init__(self):
        self._data_repo = {}
        
    def add_tfidf_data(self, train_sentence, test_sentence, y_train, label_cols, compatible_models, 
                       word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
        """
        OUTDATED!
        Params:
            train_sentence: pd.Series. Usually the sentence column of a dataframe.
                e.g. train['comment_text]
            test_sentence: pd.Series. Usually the sentence column of a dataframe.
                e.g. test['comment_text]
            y_train: pd df, with columns names = label_cols
            label_cols: list of str. label column names
            compatible_models: list of ModelName. The intented models that will use this dataset.
                e.g. [ModelName.LGB, ModelName.LOGREG]
    
            word_ngram, word_max, word_min_df, word_max_df, char_ngram, char_max, char_min_df, char_max_df: tdidf params
            
        """
        # although label_cols can be extracted from y_train, including it in params can 
        # help make sure y_train is in right format. 
        assert len(list(y_train.columns)) == len(label_cols)
        assert set(list(y_train.columns)) - set(label_cols) == set()
        x_train, x_test, data_id = tfidf_data_process(train_sentence, test_sentence, 
                                                    word_ngram=(1,1), word_max=30000)
        self.add_data(data_id, x_train, x_test, y_train, label_cols, compatible_models)
        print('{} is added to the base layer data repo'.format(data_id))
    
    def add_data(self, data_id, x_train, x_test, y_train, label_cols, compatible_models, rnn_data=False):
        """
        Params:
            x_train, x_test: dataframe
            # x_train, x_test: nparray. use .values.reshape(-1,1) to convert pd.Series to nparray
            y_train: pd df, with columns names = label columns
            label_cols: list of str. label column names
            compatible_models: list of ModelName. The intented models that will use this dataset.
                e.g. [ModelName.LGB, ModelName.LOGREG]
            rnn_data: Boolean. Whether this data is for RNN
        """
        data_dict = {
            'data_id': data_id,
            'x_train': x_train,
            'x_test': x_test,
            'labes_cols': label_cols,
            'compatible_models': set(compatible_models)
        }

        if rnn_data: 
            data_dict['y_train'] = y_train # here y_train is a df
        else:
            # label_dict = {}
            # for col in label_cols:
            #     label_dict[col] = y_train[col]
            data_dict['y_train'] = y_train.to_dict('list')
            # hence data_dict['y_train'] will be a dict with labels as keys

        self._data_repo[data_id] = data_dict

    def get_data(self, data_id):
        return self._data_repo[data_id]
    
    def remove_data(self, data_id):
        self._data_repo.pop(data_id, None)
        
    def get_compatible_models(self, data_id):
        return self._data_repo[data_id]['compatible_models']
    
    def remove_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_models'].discard(model_name)
    
    def add_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_models'].add(model_name)
                  
    def get_data_by_compatible_model(self, model_name):
        data_to_return = []
        for data_id in self._data_repo.keys():
            data = self._data_repo[data_id]
            if model_name in data['compatible_models']:
                data_to_return.append(data)
        return data_to_return
    
    def __len__(self):
        return len(self._data_repo)
    
    def __str__(self):
        output = ''
        for data_id in self._data_repo.keys():
            output+='data_id: {:20} \n\tx_train: {}\tx_test: {}\n\ty_train type: {}\n\tcompatible_models: {}\n '\
            .format(data_id, self._data_repo[data_id]['x_train'].shape, \
                    self._data_repo[data_id]['x_test'].shape, \
                    type(self._data_repo[data_id]['y_train']), \
                    self._data_repo[data_id]['compatible_models'])
        return output
    
    
def tfidf_data_process(train_sentence, test_sentence, word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
    """
    OUTDATED!
    Params:
        train_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. train['comment_text]
        test_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. test['comment_text]
        
        word_ngram, word_max, word_min_df, word_max_df, char_ngram, char_max, char_min_df, char_max_df: tdidf params

    return :x_train: sparse matrix
            y_train: DataFrame (containing all label columns)
            x_test: sparse matrix
            data_id: str, represents params
    """ 
    data_id = 'tfidf_word_{}_{}_{}_{}'.format(word_ngram, word_max, word_min_df, word_max_df)
    
    word_vectorizer = TfidfVectorizer(ngram_range=word_ngram, #1,3
                                        strip_accents='unicode',
                                        max_features=word_max,
                                        min_df = word_min_df,
                                        max_df = word_max_df,
                                        analyzer='word',
                                        stop_words='english',
                                        sublinear_tf=True,
                                        token_pattern=r'\w{1,}')
    print('fitting word')
    word_vectorizer.fit(train_sentence.values)
    print('transforming train word')
    train_word = word_vectorizer.transform(train_sentence.values)
    print('transforming test word')
    test_word = word_vectorizer.transform(test_sentence.values)

    if char_ngram == (0,0):
        print('tfidf(word level) done')
        return (train_word, test_word, data_id)

    else:
        data_id = '{}_char_{}_{}_{}_{}'.format(data_id, char_ngram, char_max, char_min_df, char_max_df)

        char_vectorizer = TfidfVectorizer(ngram_range=char_ngram,  #2,5
                                          strip_accents='unicode',
                                          max_features=char_max, #200000
                                          min_df = char_min_df,
                                          max_df = char_max_df,
                                          analyzer='char',
                                          sublinear_tf=True)

        print('fitting char')
        char_vectorizer.fit()#train_sentence_retain_punctuation.values)
        print('transforming train char')
        train_char = char_vectorizer.transform(train_sentence.values)
        print('transforming test char')
        test_char = char_vectorizer.transform(test_sentence.values)

        x_train = hstack((train_char, train_word), format='csr')
        x_test = hstack((test_char, test_word), format='csr')

        print('tfidf(word & char level) done')
        return (x_train, x_test, data_id)
    
    
def save_obj(obj, name, filepath):
    with open(filepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, filepath):
    with open(filepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class BaseLayerResultsRepo:
    def __init__(self, label_cols, filepath, load_from_file):
        """
        To start a new repo, set load_from_file to False, and give it a valid filepath so that files can be saved
        To load a save repo, set load_from_file to True, and give it the filepath
        Params:
            label_cols: list of labels
                e.g. multi-classes: ['toxic', 'severe_toxic', 'obscene'] or one-class: ['label']
        """
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self._layer1_oof_train = {}
        self._layer1_oof_test = {}
        for label in label_cols:
            self._layer1_oof_train[label] = []
            self._layer1_oof_test[label] = []
        self._base_layer_est_preds = {}
        self._model_data_id_list = []
        self._base_layer_est_scores = {}
        self._status_report = {}
        self._label_cols = label_cols
        self.filepath = filepath
        self._save_lock = False # will be set to True if remove() is invoked successfully
        if load_from_file:
            self.logger.debug('load StackNet saves from file')
            self._layer1_oof_train = load_obj('models_layer1_oof_train', self.filepath)
            self._layer1_oof_test = load_obj('models_layer1_oof_test', self.filepath)
            self._base_layer_est_preds = load_obj('models_base_layer_est_preds', self.filepath)
            self._model_data_id_list = load_obj('models_model_data_id_list', self.filepath)
            self._base_layer_est_scores = load_obj('models_base_layer_est_scores', self.filepath)
            self._status_report = load_obj('status_report', self.filepath)

    def get_model_data_id_list(self):
        return self._model_data_id_list
    
    def add(self, layer1_oof_train, layer1_oof_test, base_layer_est_preds, layer1_cv_score, model_data_id_list):
        assert type(layer1_oof_train) == dict
        assert len(list(layer1_oof_train)) == len(self._label_cols)
        assert set(list(layer1_oof_train)) - set(self._label_cols) == set()
        assert type(layer1_oof_test) == dict
        assert len(list(layer1_oof_test)) == len(self._label_cols)
        assert set(list(layer1_oof_test)) - set(self._label_cols) == set()
        for label in self._label_cols:
            assert len(layer1_oof_train[label]) == len(layer1_oof_test[label]) == len(list(base_layer_est_preds))
        assert type(base_layer_est_preds) == dict
        assert type(model_data_id_list) == list
        assert set(list(base_layer_est_preds)) - set(model_data_id_list) == set()
        assert set(list(layer1_cv_score)) - set(model_data_id_list) == set()
        for model_data_id in model_data_id_list:
            if model_data_id in set(self._model_data_id_list):
                raise ValueError('{} is already in the repo'.format(model_data_id))
        for model_data_id in model_data_id_list:
            if model_data_id not in set(self._model_data_id_list):
                self._model_data_id_list.append(model_data_id)
                self._base_layer_est_scores[model_data_id] = 0
                self._status_report[model_data_id] = {}
        for (key, values) in base_layer_est_preds.items():
            self._base_layer_est_preds[key] = values
        for label in self._label_cols:
            self._layer1_oof_train[label] += layer1_oof_train[label]
            self._layer1_oof_test[label] += layer1_oof_test[label]
        for (model_data_id, cv_score) in layer1_cv_score.items():
            self.update_report(model_data_id, 'oof_cv_score', cv_score)

    def update_report(self, model_data_id, report_key, report_value):
        if model_data_id not in set(self._model_data_id_list):
            raise ValueError('model_data_id is not found in the repo. function [add] needs '
                             'to be run first so that this model_data is in the repo')
        self._status_report[model_data_id][report_key] = report_value
        self.logger.info('StackNet report updated: {}: {} => {}'.format(model_data_id, report_key, report_value))

    def get_report(self, as_df=True):
        """
        :param as_df: boolean. True: return df. False: return dict
        :return: Return report (and convert it from dict to df if as_df is true)
        """
        if as_df:
            return pd.DataFrame.from_dict(self._status_report, orient='index')\
                .reset_index().rename(columns={'index': 'model_data'})
        else:
            return self._status_report

    def add_score(self, model_data_id, score):
        assert 0 <= score <= 1
        if model_data_id not in set(self._model_data_id_list):
            raise ValueError('{} not in the repo. please add it first'.format(model_data_id))
        if model_data_id in set(self._model_data_id_list):
            self.logger.info('{} found in repo. Update score from {} to {:.5f}'
                              .format(model_data_id, self._base_layer_est_scores[model_data_id], score))
        self._base_layer_est_scores[model_data_id] = score

    def show_scores(self):
        """
        Returns:
            list of (name, score) tuple in sorted order by score
        """
        sorted_list_from_dict = sorted(self._base_layer_est_scores.items(), key=lambda x:x[1], reverse=True)
        # for key, value in sorted_list_from_dict:
        #     print('{}\t{}'.format(value, key))
        return sorted_list_from_dict

    def get_results(self, chosen_from_layer, threshold=None, chosen_ones=None):
        """
        Params:
            chosen_from_layer: 'layer1', 'layer2'
            threshold: if not None, then return only ones in specified [layer] with score >= threshold
            chosen_ones: list of model_data_id. ignores parameter: layer
            Note:
                1. threshold and chosen_ones can NOT both have value
                2. if threshold and chosen_ones are both None, return all
                3. threshold based on:
                    layer1: gs_val_metric (e.g. gs_val_auc)
                    layer2: oof_cv_score
        Returns: 
            chosen_layer_oof_train, chosen_layer_oof_test,
            chosen_layer_est_preds, chosen_model_data_list
        """
        if threshold is not None and chosen_ones is not None:
            raise ValueError('threshold and chosen_ones can NOT both be not-None')
        if threshold is None and chosen_ones is None:
            return self._layer1_oof_train, self._layer1_oof_test, self._base_layer_est_preds
        else:
            layer1_oof_train_temp = copy.deepcopy(self._layer1_oof_train)  # copy only keep the keys, not the value reference
            layer1_oof_test_temp = copy.deepcopy(self._layer1_oof_test)  # deepcopy also keep the value reference
            base_layer_est_preds_temp = self._base_layer_est_preds.copy()
            base_layer_est_scores_temp = self._base_layer_est_scores.copy()
            status_report = self._status_report.copy()
            model_data_id_list_temp = self._model_data_id_list.copy()
            if threshold is not None:
                assert 0 <= threshold <= 1
                for (key, value) in base_layer_est_scores_temp.items():
                    if value < threshold or chosen_from_layer not in key:
                        # e.g. 'layer1' not in '1530415817__LogisticRegression_layer2'
                        self.remove(key)
            else:  # chosen_ones is not None
                assert type(chosen_ones) == list
                for model_data_id in model_data_id_list_temp:
                    if model_data_id not in chosen_ones:
                        self.remove(model_data_id)
                
            self._save_lock = False  # not actually removed, so set it back to True

            r1 = self._layer1_oof_train
            r2 = self._layer1_oof_test
            r3 = self._base_layer_est_preds
            r4 = self._model_data_id_list
            self.logger.info('chosen for layer2: {}'.format(r4))

            self._layer1_oof_train = layer1_oof_train_temp
            self._layer1_oof_test = layer1_oof_test_temp
            self._base_layer_est_preds = base_layer_est_preds_temp
            self._base_layer_est_scores = base_layer_est_scores_temp
            self._status_report = status_report
            self._model_data_id_list = model_data_id_list_temp
            return r1, r2, r3, r4
    
    def remove(self, model_data_id):
        mdid_index = self._model_data_id_list.index(model_data_id)
        self._model_data_id_list.pop(mdid_index)
        self._base_layer_est_preds.pop(model_data_id)
        self._base_layer_est_scores.pop(model_data_id)
        self._status_report.pop(model_data_id)
        for label in self._label_cols:
            self._layer1_oof_train[label].pop(mdid_index)
            self._layer1_oof_test[label].pop(mdid_index)
        self._save_lock = True
            
    def unlock_save(self):
        self._save_lock = False
            
    def save(self):
        if self._save_lock:
            self.logger.warning('save function is locked due to some results removed from the repo. '
                                'If you are sure about these changes, call unlock_save() to unlock '
                                'the save function and save again.')
        else:
            save_obj(self._model_data_id_list, 'models_model_data_id_list', self.filepath)
            save_obj(self._layer1_oof_train, 'models_layer1_oof_train', self.filepath)
            save_obj(self._layer1_oof_test, 'models_layer1_oof_test', self.filepath)
            save_obj(self._base_layer_est_preds, 'models_base_layer_est_preds', self.filepath)
            save_obj(self._base_layer_est_scores, 'models_base_layer_est_scores', self.filepath)
            save_obj(self._status_report, 'status_report', self.filepath)
            self.logger.info('StackNet data saved for: {}'.format(self._model_data_id_list))
            stacknet_report_df = self.get_report()
            stacknet_report_file = self.filepath + 'stacknet_report.csv'
            stacknet_report_df.to_csv(stacknet_report_file, index=False)
            self.logger.info('StackNet report saved at {}'.format(stacknet_report_file))


def get_oof(clf, x_train, y_train, x_test, nfolds, stratified=False, shuffle=True, seed=1001, metrics_callback=None):
    """
    Params:
        x_train, y_train, x_test
        x_train: np.ndarray of shape (X, )
        y_train: if a list, it will be converted to np.ndarray and its shape will be (Y, )
        metrics_callback: function(y_true, y_pred) to calculate per fold metrics
    """
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_kf = np.empty((nfolds, ntest)) 
    if stratified:
        kf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=seed)
    else:
        kf = KFold(n_splits=nfolds, shuffle=shuffle, random_state=seed)

    cv_score = 0
    for i, (tr_index, te_index) in enumerate(kf.split(x_train, y_train)):
        if isinstance(x_train, pd.DataFrame):  # if type(x_train).__name__ == 'DataFrame':
            x_tr, x_te = x_train.iloc[tr_index], x_train.iloc[te_index]
        else:
            x_tr, x_te = x_train[tr_index], x_train[te_index]
            try:
                clf_name = clf.clf.__name__
            except:
                clf_name = type(clf).__name__
            module_logger.warning('warning: x_train is not dataframe, '
                                  'you should NOT use models like LGB and NN where categorical_feature is needed '
                                  '(you are using {})'.format(clf_name))
        # y_tr, y_te = y_train.iloc[tr_index], y_train.iloc[te_index]
        y_train = np.array(y_train)
        y_tr, y_te = y_train[tr_index], y_train[te_index]

        module_logger.info('processing fold {} of {}...'.format(i+1, nfolds))
        if type(clf).__name__ == NNBLE.__name__:  # isinstance(model, NNBLE) not working...
            clf.train(x_tr, y_tr, x_te, y_te, x_test)
            y_pred_of_the_fold = clf.predict('x_te')
            y_pred_of_test = clf.predict('x_test')
        else:
            clf.train(x_tr, y_tr)
            y_pred_of_the_fold = clf.predict(x_te)
            y_pred_of_test = clf.predict(x_test)

        y_pred_of_the_fold = np.array(y_pred_of_the_fold).reshape(-1,)
        oof_train[te_index] = y_pred_of_the_fold
        if metrics_callback is not None:
            score = metrics_callback(y_te, y_pred_of_the_fold)
            module_logger.info('metric of fold {}: {}'.format(i+1, score))
            cv_score += score

        y_pred_of_test = np.array(y_pred_of_test).reshape(-1,)
        oof_test_kf[i, :] = y_pred_of_test

    cv_score = cv_score / nfolds

    oof_test[:] = oof_test_kf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), cv_score#, oof_test_kf.reshape(-1, nfolds)


def compute_layer1_oof(bldr, model_pool, label_cols, nfolds=5, seed=2018, sfm_threshold=None, metrics_callback=None):
    """
    Params:
        bldr: an instance of BaseLayerDataRepo
        model_pool: dict. key: an option from ModelName. value: A model
        label_cols: list. names of labels
        nfolds: int 
        seed: int. for reproduce purpose
        sfm_threshold: str. e.g. 'median', '2*median'. if not None, then use SelectFromModel to select features
            that have importance > sfm_threshold
        metrics_callback: function(y_true, y_pred) to calculate per fold metrics
    Returns:
        layer1_est_preds: This is the prediction of the layer 1 model_data, you can submit it to see the LB score
        layer1_oof_train: This will be used as training features in higher layers (one from each model_data)
        layer1_oof_mean_test: This will be used as testing features in higher layers (one from each model_data)
        layer1_cv_score: This is a list of cv scores (e.g. auc) of all layer 2 model_data
        model_data_id_list: This is the list of all layer 1 model_data
    """
    layer1_est_preds = {} # directly preditions from the base layer estimators # also layer1_oof_nofold_test

    layer1_oof_train = {}
    layer1_oof_mean_test = {}
    layer1_cv_score = {}
    #layer1_oof_perfold_test = {}
    #layer1_oof_nofold_test = {}

    model_data_id_list = []

    for i, label in enumerate(label_cols):
    #     layer1_oof_train[label] = []
    #     layer1_oof_test[label] = []
        for model_name in model_pool.keys():
            # model_name is AwgE__LGB, so model_id will be LGB, not the number
            model_id = model_name.split('__')[1]
            for data in bldr.get_data_by_compatible_model(model_id):

                model_data_id = '{}_{}_{}'.format(model_name, data['data_id'], 'layer1')
                current_run = 'label: {:8s} model_data_id: {}'.format(label, model_data_id)
                module_logger.info('StackNet layer1: '+current_run)

                x_train = data['x_train']  # x_train: dataframe
                y_train = data['y_train'][label]  # y_train: list
                x_test = data['x_test']

                if sfm_threshold is not None:
                    raise ValueError('sfm not implemented yet')
                    # after perform this, dataframe is converted to np.ndarray, so 'category'
                    # type column info is lost.
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.feature_selection import SelectFromModel
                    model = LogisticRegression(solver='sag')
                    sfm = SelectFromModel(model, threshold=sfm_threshold)
                    print('dimension before selecting: train:{} test:{}'.format(x_train.shape, x_test.shape))
                    x_train = sfm.fit_transform(x_train, y_train)
                    x_test = sfm.transform(x_test)
                    print('dimension after selecting: train:{} test:{}'.format(x_train.shape, x_test.shape))
                else:
                    x_train = x_train#.values # here df is NOT converted to np array

                model = model_pool[model_name]
                if 'PERLABEL' in str(model_name):
                    model.set_params_for_label(label)

                if nfolds != 0:
                    oof_train, oof_mean_test, cv_score = \
                        get_oof(model, x_train, y_train, x_test, nfolds=nfolds,
                                stratified=True, seed=seed, metrics_callback=metrics_callback)
                    module_logger.info('oof is done')
                else:
                    raise ValueError('nfolds of oof can NOT be 0!')

                module_logger.info('Training using all data and gen prediction for submission...')
                if type(model).__name__ == NNBLE.__name__:  # isinstance(model, NNBLE) not working...
                    model.train(x_train, y_train, None, None, x_test)
                    est_preds = model.predict('x_test')
                else:
                    model.train(x_train, y_train)
                    est_preds = model.predict(x_test)
                est_preds = np.array(est_preds).reshape(-1,)

                if label not in layer1_oof_train:
                    layer1_oof_train[label] = []
                    layer1_oof_mean_test[label] = []
                    #layer1_oof_perfold_test[label] = []
                    #layer1_oof_nofold_test[label] = []
                layer1_oof_train[label].append(oof_train)
                layer1_oof_mean_test[label].append(oof_mean_test)
                #layer1_oof_perfold_test[label].append(oof_perfold_test)
                #layer1_oof_nofold_test[label].append(est_preds.reshape(-1,1))

                if model_data_id not in layer1_est_preds:
                    layer1_est_preds[model_data_id] = np.empty((x_test.shape[0],len(label_cols)))
                    model_data_id_list.append(model_data_id)
                layer1_est_preds[model_data_id][:,i] = est_preds
                layer1_cv_score[model_data_id] = cv_score  # TODO: unlike others, here assuming one label

    return layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, layer1_cv_score, model_data_id_list


def combine_layer_oof_per_label(layer1_oof_dict, label):
    """
    Util method for stacking
    """
    x = None
    data_list = layer1_oof_dict[label]
    for i in range(len(data_list)):
        if i == 0:
            x = data_list[0]
        else:
            x = np.concatenate((x, data_list[i]), axis=1)
    return x


def compute_layer2_oof(model_pool, layer2_inputs, train, label_cols, nfolds, seed, auto_sub_func,
                       metric, metrics_callback=None):
    """
    Params:
        model_pool: dict. key: an option from ModelName. value: A model
        layer2_inputs: dict. key: an option from ModelName. value: chosen results from an instance of BaseLayerDataRepo
        train: pd. training data is required here to extract labels from it. For now, please make sure the train is not shuffled
            (TODO: THIS SHOULD BE INCLUDED IN layer2_inputs, which in terms should be included in BaseLayerDataRepo.)
        label_cols: list. names of labels
        nfolds: int
        seed: int. for reproduce purpose
    Returns:
        layer2_est_preds: This is the prediction of the layer 2 model_data, you can submit it to see the LB score
        layer2_oof_train: This will be used as training features in higher layers (one from each model_data)
        layer2_oof_mean_test: This will be used as testing features in higher layers (one from each model_data)
        layer2_cv_score: This is a list of cv scores (e.g. auc) of all layer 2 model_data
        layer2_model_data_list: This is the list of all layer 2 model_data
    """
    layer2_est_preds = {} # directly preditions from the base layer estimators

    layer2_oof_train = {}
    layer2_oof_test = {}
    layer2_cv_score = {}

    layer2_model_data_list = []

    for model_name in model_pool.keys():
        module_logger.info('Generating Layer2 model {} OOF'.format(model_name))
        for i, label in enumerate(label_cols):
            #assert train.shape[0] == 159571

            model = model_pool[model_name]

            layer1_oof_train_loaded, layer1_oof_test_loaded, _ = layer2_inputs[model_name]

            x_train = combine_layer_oof_per_label(layer1_oof_train_loaded, label)
            x_test = combine_layer_oof_per_label(layer1_oof_test_loaded, label)

            oof_train, oof_test, cv_score = get_oof(model,  x_train, train[label], x_test,
                                                    nfolds, seed, metrics_callback=metrics_callback)

            if label not in layer2_oof_train:
                layer2_oof_train[label] = []
                layer2_oof_test[label] = []
            layer2_oof_train[label].append(oof_train)
            layer2_oof_test[label].append(oof_test)

            model_id = '{}_{}'.format(model_name, 'layer2')
            if type(model).__name__ == NNBLE.__name__:  # isinstance(model, NNBLE) not working...
                model.train(x_train, train[label], None, None, x_test)
                est_preds = model.predict('x_test')
            else:
                model.train(x_train, train[label])
                est_preds = model.predict(x_test)
            est_preds = np.array(est_preds).reshape(-1, )
            if auto_sub_func is not None:
                try:
                    auto_sub_func(est_preds, model_id)
                except:
                    print('Auto Submission Failed: ', sys.exc_info()[0])

            if model_id not in layer2_est_preds:
                layer2_est_preds[model_id] = np.empty((x_test.shape[0], len(label_cols)))
                layer2_model_data_list.append(model_id)
            layer2_est_preds[model_id][:, i] = est_preds
            layer2_cv_score[model_id] = cv_score  # TODO: unlike others, here assuming one label

    return layer2_est_preds, layer2_oof_train, layer2_oof_test, layer2_cv_score, layer2_model_data_list