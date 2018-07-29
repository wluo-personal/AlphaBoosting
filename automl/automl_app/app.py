import pandas as pd
import numpy as np
import os, json, gc, logging, shutil, pickle, time
from automl_libs import utils, grid_search, grid_search_v2, nn_libs
from automl_libs import stacknet_v2 as stacknet
# from automl_libs import stacknet
from enum import Enum
import pdb


class AlphaBoosting:

    class Stage(Enum):
        FEATURE_ENGINEERING = 1
        VALIDATION_DOWNSAMPLING_GEN_INDEX = 2
        VALIDATION_DOWNSAMPLING_SPLIT = 3
        VALIDATION_DOWNSAMPLING_GEN = 4
        CONCAT_DATA = 5
        # GRID_SEARCH = 6
        # STACKNET = 7

    def __init__(self, config_file, features_to_gen, params_gen, auto_sub_func=None):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.logger.info('='*10+'START'+'='*10)

        if config_file is None:
            raise Exception('config file can not be None')
        if not os.path.exists(config_file):
            raise Exception('config file can not be found')
        self.config_dict = json.load(open(config_file, 'r'))
        self.ROOT = self.config_dict['project_root']
               
        # 1. run_record need to to provides so that the previous run(if there is one) info can
        # be loaded, which will determine what need to be rerun and what don't
        # 2. don't create this file or modify this file
            
        self.features_to_gen = features_to_gen
        self.params_gen = params_gen
        self.auto_sub_func = auto_sub_func

        self.OUTDIR = self.config_dict['project_root'] + 'output/'
        self.TEMP_DATADIR = self.config_dict['project_root'] + 'temp_data/'
        self.run_record_url = self.TEMP_DATADIR + self.config_dict['last_run_record_filename']
        self.train_data_url = self.config_dict['train_data_url']
        self.test_data_url = self.config_dict['test_data_url']
        self.label = self.config_dict['label']
        self.down_sampling_amt = self.config_dict['down_sampling_amt']
        self.down_sampling_ratio = self.config_dict['down_sampling_ratio'] 
        
        # read data and determine validation set
        self._read_data()
        if self.config_dict['validation_index'] is not None:
            self.validation_index = pickle.load(open(self.config_dict['validation_index'], 'rb'))
            self.validation_ratio = None # means the 'ratio' param is overruled
        else:
            try:
                val_ratio = self.config_dict['validation_ratio']
            except KeyError:
                raise ValueError('since validation_index is null in config file, validation_ratio must be provided')
            self.validation_ratio = val_ratio
            self.validation_index = list(range(int(self.train_len*(1-self.validation_ratio)), self.train_len))
            
        downsampling_amount_changed = False
        down_sampling_ratio_changed = False
        val_index_changed = False
        
        if not os.path.exists(self.run_record_url):
            self.logger.info('Run record file [{}] not found. Begin the first time run...'.format(self.run_record_url))
        else:
            self.logger.info('Run record file [{}] found. Continue from the previous run...'.format(self.run_record_url))
            with open(self.run_record_url, 'r') as f: run_record_dict = json.load(f)
            
            # check if validation and down sampling need to be redone
            prev_down_sampling_amt = run_record_dict['down_sampling_amt']
            if prev_down_sampling_amt != self.down_sampling_amt:
                self.logger.debug('Down sampling amount: previous: {}. new: {}'
                                  .format(prev_down_sampling_amt, self.down_sampling_amt))
                downsampling_amount_changed = True
            
            prev_down_sampling_ratio = run_record_dict['down_sampling_ratio']
            if self.down_sampling_ratio != prev_down_sampling_ratio:
                self.logger.debug('Down sampling ratio: previous: {}. new: {}'
                                  .format(prev_down_sampling_ratio, self.down_sampling_ratio))
                down_sampling_ratio_changed = True
                
            prev_validation_index = pickle.load(open(run_record_dict['validation_index'],'rb'))
            if self.validation_index != prev_validation_index:
                self.logger.debug('Validation index: previous: {}... new: {}...'
                                  .format(prev_validation_index[:3], self.validation_index[:3]))
                val_index_changed = True

        # build relavent directories
        self.FEATUREDIR = self.TEMP_DATADIR + 'features/'
        if not os.path.exists(self.OUTDIR): os.makedirs(self.OUTDIR)
        if not os.path.exists(self.TEMP_DATADIR): os.makedirs(self.TEMP_DATADIR)
        if not os.path.exists(self.FEATUREDIR): os.makedirs(self.FEATUREDIR)
            
        # generate todo list: c
        self.logger.info('generate todo list')
        to_do_dict = self._generate_todo_list()
        
        if downsampling_amount_changed or down_sampling_ratio_changed or val_index_changed:
            to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_GEN_INDEX.name] = False
            to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_SPLIT.name] = False
            to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_GEN.name] = False
            
            if os.path.exists(self.TEMP_DATADIR + 'down_sampling_idx.pkl'): os.remove(self.TEMP_DATADIR + 'down_sampling_idx.pkl')
            if os.path.exists(self.TEMP_DATADIR + 'val.pkl'): os.remove(self.TEMP_DATADIR + 'val.pkl')
            for i in range(prev_down_sampling_amt): 
                if os.path.exists(self.TEMP_DATADIR + str(i) + '.pkl'):
                    os.remove(self.TEMP_DATADIR + str(i) + '.pkl')
            shutil.rmtree(self.TEMP_DATADIR + 'split/')

        # feature engineering
        self.logger.info('STAGE: ' + self.Stage.FEATURE_ENGINEERING.name)
        self._feature_engineering(to_do_dict)
        
        # get validation
        self.logger.info('STAGE: ' + self.Stage.VALIDATION_DOWNSAMPLING_GEN.name)
        self._validation_and_down_sampling(to_do_dict)
        
        # concatenant test: c
        self.logger.info('STAGE: ' + self.Stage.CONCAT_DATA.name)
        self._concat_test(to_do_dict)
        
        # grid search
        if self.config_dict['do_gs']:
            self.logger.info('GRID SEARCH: Perform')
            self._grid_search()
        else:
            self.logger.info('GRID SEARCH: Skip')

        # grid search
        if self.config_dict['do_stacknet']:
            self.logger.info('STACK NET: Perform')
            self._stacknet()
        else:
            self.logger.info('STACK NET: Skip')

        # save self.run_record_url:
        self.logger.info('save run record')
        self._save_run_record()

    # util functions
    def _read_data(self):
        self.train = pd.read_pickle(self.train_data_url)
        self.test = pd.read_pickle(self.test_data_url)
        self.df = pd.concat([self.train, self.test], sort=True, ignore_index=True)
        self.train_len = self.train.shape[0]

    @staticmethod
    def _renew_status(to_do_dict, key, file_url):
        to_do_dict[key] = True
        with open(file_url, 'w') as f:
            json.dump(to_do_dict, f, indent=4, sort_keys=True)

    def _save_run_record(self):
        run_record = {
            'project_root':         self.ROOT,
            'train_data_url':       self.train_data_url,
            'test_data_url':        self.test_data_url,
            'label':                self.label,
            'down_sampling_amt':    self.down_sampling_amt,
            'down_sampling_ratio':  self.down_sampling_ratio
        }
        if self.validation_ratio is not None:
            # 'not None' means param: [validation_ratio] is not overruled by param [validation_index]
            run_record['validation_ratio'] = self.validation_ratio

        val_index_url = self.TEMP_DATADIR + 'val_index.pkl'
        run_record['validation_index'] = val_index_url 
        pickle.dump(self.validation_index, open(val_index_url,'wb'))
        self.logger.info('val index is saved at {}'.format(val_index_url))
        with open(self.run_record_url, 'w') as f: json.dump(run_record, f, indent=4, sort_keys=True)
        self.logger.info('run record is saved at {}'.format(self.run_record_url))
        del run_record 
        gc.collect()

    @staticmethod
    def _get_file_concat(base_df, split_folder, concat_folder, is_train, file_name_body):
        prefix = 'train' if is_train else 'test'
        file_name = file_name_body + '.pkl'
        for file in os.listdir(split_folder):
            inter_split = file.split('.')
            if inter_split[-1] == 'pkl':
                splitted = inter_split[0].split('__')
                if splitted[0] == prefix:
                    tmp_pkl = pd.read_pickle(split_folder + file)
                    base_df['__'.join(splitted[1:])] = tmp_pkl
                    del tmp_pkl
                    gc.collect()
        base_df.reset_index(drop=True).to_pickle(concat_folder + file_name)
        del base_df
        gc.collect()
            
    # main functions
    def _generate_todo_list(self):
        if os.path.exists(self.TEMP_DATADIR + 'todo_list.json'):
            with open(self.TEMP_DATADIR + 'todo_list.json', 'r') as file:
                to_do_dict = json.load(file)
        else:
            to_do_dict = {s.name: False for s in self.Stage}
            with open(self.TEMP_DATADIR + 'todo_list.json', 'w') as file:
                json.dump(to_do_dict, file, indent=4, sort_keys=True)
        return to_do_dict
    
    def _feature_engineering(self, to_do_dict):
        stage = self.Stage.FEATURE_ENGINEERING.name
        if not to_do_dict[stage]:
            for feature_to_gen in self.features_to_gen:
                self._add_column(feature_to_gen)
        self._renew_status(to_do_dict, stage, (self.TEMP_DATADIR + 'todo_list.json'))
    
    def _validation_and_down_sampling(self, to_do_dict):
        split_folder = []
        index = []
        if not to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_GEN.name]:
            if self.down_sampling_amt != 0:
                # down sampling
                down_sampling_url = self.TEMP_DATADIR + 'split/'
                if not os.path.exists(down_sampling_url):
                    os.makedirs(down_sampling_url)
                index.extend(self._generate_down_sampling_index_file(
                    to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_GEN_INDEX.name]))
                for i in range(self.down_sampling_amt): 
                    split_folder.append(down_sampling_url+str(i)+'/')
                    if not os.path.exists(split_folder[-1]): os.makedirs(split_folder[-1])

            # validation
            split_folder.append(self.TEMP_DATADIR + 'split/val/')
            index.append(self.validation_index)
            if not os.path.exists(split_folder[-1]):
                os.makedirs(split_folder[-1])
                
            self._renew_status(to_do_dict, self.Stage.VALIDATION_DOWNSAMPLING_GEN_INDEX.name,
                               self.TEMP_DATADIR + 'todo_list.json')
        
        # split files
        if not to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_SPLIT.name]:
            for file in os.listdir(self.FEATUREDIR):
                split_file = file.split('.')
                if split_file[-1] == 'pkl':
                    splitted = split_file[0].split('__')
                    if splitted[0] == 'train':
                        for i in range(len(index)):
                            if not os.path.exists(split_folder[i] + file):
                                tmp_df = pd.read_pickle(self.FEATUREDIR + file)
                                tmp_df.loc[index[i]].reset_index(drop=True).to_pickle(split_folder[i] + file)
                                del tmp_df
                                gc.collect()
            self._renew_status(to_do_dict, self.Stage.VALIDATION_DOWNSAMPLING_SPLIT.name, self.TEMP_DATADIR + 'todo_list.json')
        
        # concat files
        if not to_do_dict[self.Stage.VALIDATION_DOWNSAMPLING_GEN.name]:
            if self.down_sampling_amt == 0:
                index.append(sorted(list(set(range(self.train_len)).difference(set(self.validation_index)))))
                split_folder.append(self.FEATUREDIR)
            for i in range(len(split_folder)):
                file_name_body = 'train' if split_folder[i] == self.FEATUREDIR else split_folder[i].split('/')[-2]
                self._get_file_concat(base_df=self.train.loc[index[i]].copy().reset_index(drop=True),
                                      split_folder=split_folder[i], 
                                      concat_folder=self.TEMP_DATADIR, 
                                      is_train=True, 
                                      file_name_body=file_name_body)
            self._renew_status(to_do_dict, self.Stage.VALIDATION_DOWNSAMPLING_GEN.name, self.TEMP_DATADIR + 'todo_list.json')

    def _grid_search(self):
        # stage = self.Stage.GRID_SEARCH.name
        # if not to_do_dict[stage]:
        data_name, train, val, test, y_test, categorical_features, feature_cols, label_col = self._get_final_data()
        X_train = train[feature_cols]
        y_train = train[label_col]
        X_val = val[feature_cols]
        y_val = val[label_col]
        X_test = test[feature_cols]

        gs_models = self.config_dict['gs_models']
        gs_record_dir = self.OUTDIR
        gs_search_rounds = self.config_dict['gs_search_rounds']
        gs_cv = self.config_dict['gs_cv']
        gs_nfold = self.config_dict['gs_nfold']
        gs_verbose_eval = self.config_dict['gs_verbose_eval']
        gs_do_preds = self.config_dict['gs_do_preds']
        gs_sup_warning = self.config_dict['gs_suppress_warning']

        grid_search_v2.gs(data_name, X_train, y_train, X_val, y_val,
                       categorical_features, search_rounds=gs_search_rounds,
                       gs_record_dir=gs_record_dir,
                       gs_params_gen=self.params_gen, gs_models=gs_models,
                       cv=gs_cv, nfold=gs_nfold, verbose_eval=gs_verbose_eval,
                       stratified=self.config_dict['gs_cv_stratified'],
                       do_preds=gs_do_preds, X_test=X_test, y_test=y_test,
                       auto_sub_func=self.auto_sub_func,
                       # auto_sub_func=None,
                       preds_save_path=self.OUTDIR+'gs_saved_preds/',
                       suppress_warning=gs_sup_warning)
        del train, val, test; gc.collect()
        # self._renew_status(to_do_dict, stage, self.TEMP_DATADIR + 'todo_list.json')

    def _stacknet(self):
        # if not to_do_dict[self.Stage.STACKNET.name]:
        # seems need absolute path to save
        oof_path = self.OUTDIR + 'oof/'
        if not os.path.exists(oof_path):
            os.makedirs(oof_path)
        gs_result_path = self.OUTDIR
        data_name, train, val, test, y_test, categorical_features, feature_cols, label_cols = self._get_final_data()
        # convert label_cols to list so that y_train will be a dataframe, which is required stacknet layers
        if not isinstance(label_cols, list):
            label_cols = [label_cols]
        train = pd.concat([train, val])
        layers_to_built = self.config_dict['build_stacknet_layers']
        self.logger.info('layers to be built: {}'.format(layers_to_built))
        _, seed = self.params_gen('lgb')  # does not matter lgb or any other, we just want a seed
        if 1 in layers_to_built:
            stacknet.layer1(data_name, train, test, y_test, categorical_features, feature_cols, label_cols,
                            params_source=self.config_dict['params_source'],
                            build_layer1_amount=self.config_dict['build_layer1_amount'],
                            params_gen=self.params_gen, layer1_models=self.config_dict['layer1_models'],
                            top_n_by=self.config_dict['top_n_by'],
                            top_n_gs=self.config_dict['top_n_per_gs_res_for_layer1'],
                            oof_nfolds=self.config_dict['oof_nfolds_layer1'], seed=seed, oof_path=oof_path,
                            auto_sub_func=self.auto_sub_func, preds_save_path=self.OUTDIR+'sn_saved_preds/',
                            pg_save_path=self.OUTDIR, metric=self.config_dict['report_metric'],
                            gs_result_path=gs_result_path)
        if 2 in layers_to_built:
            stacknet.layer2(train, y_test, label_cols, params_gen=self.params_gen,
                            oof_path=oof_path, metric=self.config_dict['report_metric'],
                            oof_nfolds=self.config_dict['oof_nfolds_layer2'], seed=seed,
                            layer1_thresh_or_chosen=self.config_dict['layer1_thresh_or_chosen_for_layer2'],
                            layer2_models=self.config_dict['layer2_models'], auto_sub_func=self.auto_sub_func,
                            preds_save_path=self.OUTDIR+'sn_saved_preds/')

        # self._renew_status(to_do_dict, self.Stage.STACKNET.name, self.TEMP_DATADIR + 'todo_list.json')

    def _get_final_data(self):
        if self.down_sampling_amt == 0:
            train = pd.read_pickle(self.TEMP_DATADIR+'train.pkl')
        else:
            train = pd.read_pickle(self.TEMP_DATADIR+'0.pkl')
        val = pd.read_pickle(self.TEMP_DATADIR+'val.pkl')
        test = pd.read_pickle(self.TEMP_DATADIR+'test.pkl')
        y_test_url = self.config_dict['test_label_url']
        y_test = None
        if y_test_url is not None:
            y_test = np.load(y_test_url)  # could be none
            assert len(test) == len(y_test)

        not_features = self.config_dict['not_features']
        categorical_features = list(set(self.config_dict['categorical_features']) - set(not_features))
        label_col = self.config_dict['label']
        label_col_as_list=[label_col]
        feature_cols = list(set(train.columns) - set(not_features) - set(label_col_as_list))
        debug_data = self.config_dict['debug_data']
        if 0 < debug_data < 1:
            train = train.head(int(debug_data*len(train)))
            val = val.head(int(debug_data*len(val)))
            self.logger.info('DEBUG mode is on, {}% of train,val data are chosen'.format(debug_data*100))
        elif debug_data > 1:
            train = train.head(debug_data)
            val = val.head(debug_data)
            self.logger.info('DEBUG mode is on, first {} rows of train,val data are chosen'.format(debug_data))
        data_name = self.config_dict['data_name']
        self.logger.info('Data <{}> retrieved. Shape: train {} | val {} | test {} | contain test label: {} | '
                         '{} cat features | {} total features | y name: {}'
                         .format(data_name, train.shape, val.shape, test.shape, False if y_test is None else True,
                                 len(categorical_features), len(feature_cols), label_col))
        return data_name, train, val, test, y_test, categorical_features, feature_cols, label_col

        
    # support functions
    # create a feature
    def _add_column(self, feature_to_gen):
        """
        feature_to_gen: dict
            e.g:
            _____
            {
                "feature_cols": [
                    "a",
                    "b"
                ],
                "function": "count",
                "params": {
                    "coefficient": 10,
                    "col": "l",
                    "fillna": 22,
                    "n": 2,
                    "splitCol": "a",
                    "trainLen": 18
                }
            }
        """
        func = feature_to_gen.get('function')
        feature_cols = feature_to_gen.get('feature_cols')
        params = feature_to_gen.get('params')
        generated_feature_name = '__'.join([func.__name__, '_'.join(feature_cols)])
        if feature_to_gen.get('params') != {}: generated_feature_name += '__' + '_'.join(map(str, params.values()))
        params['train_len'] = self.train_len
        if not os.path.exists(self.FEATUREDIR + 'train__' + generated_feature_name + '.pkl'):
            # TODO: test if passing df=df[feature_cols+[self.label]] can save memory
            _df = func(df=self.df, cols=feature_to_gen.get('feature_cols'), dummy_col=self.label,
                       generated_feature_name=generated_feature_name, params=params)
            utils.save(df=_df, train_len=self.train_len, url=self.FEATUREDIR, name=generated_feature_name)
    
    # concat test
    def _concat_test(self, to_do_dict):
        stage = self.Stage.CONCAT_DATA.name
        if not to_do_dict[stage]:
            self._get_file_concat(base_df=self.test.copy(), 
                                  split_folder=self.FEATUREDIR, 
                                  concat_folder=self.TEMP_DATADIR, 
                                  is_train=False, 
                                  file_name_body='test')
            self._renew_status(to_do_dict, stage, self.TEMP_DATADIR + 'todo_list.json')
        gc.collect()
    
    def _generate_down_sampling_index_file(self, has_file_built):
        
        def _downsampling(positive_idx, negative_idx, ratio):
            idx = np.random.choice(negative_idx, int(ratio*len(negative_idx)), replace=False)
            idx = np.concatenate((idx, positive_idx))
            return np.sort(idx).astype(int).tolist()
        
        index = []
        down_sampling_idx_url = self.TEMP_DATADIR + 'down_sampling_idx.pkl'
        if has_file_built:
            index = pickle.load(open(down_sampling_idx_url, 'rb'))
        else:
            train_exclude_val = self.train.drop(self.validation_index, axis=0)
            positive = list(train_exclude_val[train_exclude_val[self.label]==1].index.values)
            negative = list(train_exclude_val[train_exclude_val[self.label]==0].index.values)
            ratio = len(positive) / len(negative) if self.down_sampling_ratio is None else self.down_sampling_ratio
            for i in range(self.down_sampling_amt):
                index.append(_downsampling(positive, negative, ratio))
            del train_exclude_val
            gc.collect()
            pickle.dump(index, open(down_sampling_idx_url,'wb'))
        return index

