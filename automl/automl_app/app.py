import os, json, gc, logging, shutil
import pandas as pd
import numpy as np
from automl_libs import utils

import pdb
    
class AlphaBoosting:
    def __init__(self, config_file, features_to_gen):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        if config_file is None:
            raise Exception('config file can not be None')
            if not os.path.exists(config_file):
                raise Exception('config file can not be found')
        self.config_dict = json.load(open(config_file, 'r'))
        self.ROOT = self.config_dict['project_root']
               
        # 1. run_record need to to provides so that the previous run(if there is one) info can
        # be loaded, which will determine what need to be rerun and what don't
        # 2. don't create this file or modify this file
        run_record = self.ROOT + self.config_dict['run_record_filename']
            
        self.features_to_gen = features_to_gen
               
        self.OUTDIR = self.config_dict['project_root'] + 'output/'
        self.LOGDIR = self.config_dict['project_root'] + 'log/'
        self.DATADIR = self.config_dict['project_root'] + 'data/'
        self.train_data_url = self.config_dict['train_data_url']
        self.test_data_url = self.config_dict['test_data_url']
        self.label = self.config_dict['label']
        self.down_sampling_amt = self.config_dict['down_sampling_amt']
        self.down_sampling_ratio = self.config_dict['down_sampling_ratio'] 
        
        # read data and determine validation set
        self._read_data()
        if self.config_dict['validation_index'] is not None:
            self.validation_index = json.load(open(self.config_dict['validation_index']))
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
        
        
        if not os.path.exists(run_record):
            self.logger.info('Run record file [{}] not found. Begin the first time run...'.format(run_record))
        else:
            self.logger.info('Run record file [{}] found. Continue from the previous run...'.format(run_record))
            with open(run_record, 'r') as f: run_record_dict = json.load(f)
            
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
                
            prev_validation_index = json.load(open(run_record_dict['validation_index']))
            if self.validation_index != prev_validation_index:
                self.logger.debug('Validation index: previous: {}... new: {}...'
                                  .format(prev_validation_index[:3], self.validation_index[:3]))
                val_index_changed = True
                
        
        # build relavent directories
        self.FEATUREDIR = self.DATADIR + 'features/'
        if not os.path.exists(self.OUTDIR): os.makedirs(self.OUTDIR)
        if not os.path.exists(self.LOGDIR): os.makedirs(self.LOGDIR)
        if not os.path.exists(self.DATADIR): os.makedirs(self.DATADIR)
        if not os.path.exists(self.FEATUREDIR): os.makedirs(self.FEATUREDIR)
            
        # save run_record:
        self.logger.info('save run record')
        self._save_run_record(run_record)
        
        # generate todo list: c
        self.logger.info('generate todo list')
        dictionary = self._generate_todo_list()
        
        if downsampling_amount_changed or down_sampling_ratio_changed or val_index_changed:
            dictionary['val_downsample_generate_index'] = False
            dictionary['val_downsample_split'] = False
            dictionary['val_downsample_generation'] = False
            
            if os.path.exists(self.LOGDIR + 'down_sampling_idx.json'): os.remove(self.LOGDIR + 'down_sampling_idx.json')
            if os.path.exists(self.LOGDIR + 'val.pkl'): os.remove(self.DATADIR + 'val.pkl')
            for i in range(prev_down_sampling_amt): 
                if os.path.exists(self.DATADIR + str(i) + '.pkl'):
                    os.remove(self.DATADIR + str(i) + '.pkl')
            shutil.rmtree(self.DATADIR + 'split/')
        
        # feature engineering
        self.logger.info('feature engineering')
        self._feature_engineering(dictionary)
        
        # get validation
        self.logger.info('validation')
        self._validation_and_down_sampling(dictionary)
        
        # concatenant test: c
        self.logger.info('concat test')
        self._concat_test(dictionary)
        
        # grid search
        self.logger.info('grid search')
        self._grid_search(dictionary)
    
    
    ######### util functions #########
    def _read_data(self):
        self.train = pd.read_pickle(self.train_data_url)
        self.test = pd.read_pickle(self.test_data_url)
        self.df = pd.concat([self.train, self.test], ignore_index=True)
        self.train_len = self.train.shape[0]
        
    def _renew_status(self, dictionary, key, file_url):
        dictionary[key] = True
        with open(file_url, 'w') as f:
            json.dump(dictionary, f, indent=4, sort_keys=True)

            
    def _save_run_record(self, run_record_url):
        run_record = {
            'project_root':         self.ROOT,
            'train_data_url':       self.train_data_url,
            'test_data_url':        self.test_data_url,
            'label':                self.label,
            'down_sampling_amt':    self.down_sampling_amt,
            'down_sampling_ratio':  self.down_sampling_ratio
        }
        if self.validation_ratio is not None: # means param: [validation_ratio] is not overruled by param [validation_index]
            run_record['validation_ratio'] = self.validation_ratio
            
        val_index_url = self.LOGDIR + 'val_index.json'
        run_record['validation_index'] = val_index_url 
        with open(val_index_url, 'w') as f: json.dump(self.validation_index, f, indent=4, sort_keys=True)
            
        with open(run_record_url, 'w') as f: json.dump(run_record, f, indent=4, sort_keys=True)
        del run_record 
        gc.collect()
            
    def _get_file_concat(self, base_df, split_folder, concat_folder, is_train, file_name_body):
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
            
            
    
    ######### main functions #########
    def _generate_todo_list(self):
        if os.path.exists(self.LOGDIR + 'todo_list.json'):
            with open(self.LOGDIR + 'todo_list.json', 'r') as file:
                dictionary = json.load(file)
        else:
            dictionary = {
                'feature_engineering':           False, 
                'val_downsample_generate_index': False,
                'val_downsample_split':          False,
                'val_downsample_generation':     False,
                'concat_test':                   False,
                'grid_search':                   False
            }
            with open(self.LOGDIR + 'todo_list.json', 'w') as file: 
                json.dump(dictionary, file, indent=4, sort_keys=True)
        return dictionary
    
    def _feature_engineering(self, dictionary):
        feature_engineering_file_url = self.LOGDIR + 'feature_engineering.json'
        if not dictionary['feature_engineering']:
            for feature_to_gen in self.features_to_gen:
                self._add_column(feature_to_gen)
        self._renew_status(dictionary, 'feature_engineering', (self.LOGDIR + 'todo_list.json'))
    
    def _validation_and_down_sampling(self, dictionary):
        split_folder = []
        index = []
        down_sampling_url = None
        if not dictionary['val_downsample_generation']:
            if self.down_sampling_amt != 0:
                # down sampling
                down_sampling_url = self.DATADIR + 'split/'
                if not os.path.exists(down_sampling_url): os.makedirs(down_sampling_url)
                index.extend(self._generate_down_sampling_index_file(dictionary['val_downsample_generate_index']))
                for i in range(self.down_sampling_amt): 
                    split_folder.append(down_sampling_url+str(i)+'/')
                    if not os.path.exists(split_folder[-1]): os.makedirs(split_folder[-1])

            # validation
            split_folder.append(self.DATADIR + 'split/val/')
            index.append(self.validation_index)
            if not os.path.exists(split_folder[-1]): os.makedirs(split_folder[-1])
                
            self._renew_status(dictionary, 'val_downsample_generate_index', self.LOGDIR + 'todo_list.json')
        
        # split files
        if not dictionary['val_downsample_split']:
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
            self._renew_status(dictionary, 'val_downsample_split', self.LOGDIR + 'todo_list.json')
        
        # concat files
        if not dictionary['val_downsample_generation']:
            if self.down_sampling_amt == 0:
                index.append(sorted(list(set(range(self.train_len)).difference(set(self.validation_index)))))
                split_folder.append(self.FEATUREDIR)
            for i in range(len(split_folder)):
                file_name_body = 'train' if split_folder[i] == self.FEATUREDIR else split_folder[i].split('/')[-2]
                self._get_file_concat(base_df=self.train.loc[index[i]].copy().reset_index(drop=True),
                                      split_folder=split_folder[i], 
                                      concat_folder=self.DATADIR, 
                                      is_train=True, 
                                      file_name_body=file_name_body)
            self._renew_status(dictionary, 'val_downsample_generation', self.LOGDIR + 'todo_list.json')
        
    
    def _grid_search(self, dictionary):
        if not dictionary['grid_search']:
            if self.down_sampling_amt == 0:
                train = pd.read_pickle(self.DATADIR+'train.pkl')
            else:
                train = pd.read_pickle(self.DATADIR+'0.pkl')
            val = pd.read_pickle(self.DATADIR+'val.pkl')
            test = pd.read_pickle(self.DATADIR+'test.pkl')
            
            categorical_features = self.config_dict['categorical_features']
            not_features = self.config_dict['not_features']
            label_col = self.config_dict['label']
            label_col_as_list=[label_col]
            feature_cols = list(set(train.columns) - set(not_features) - set(label_col_as_list))
            
            gs_record = self.ROOT + self.config_dict['gs_record_filename']
            gs_search_rounds = self.config_dict['gs_search_rounds']
            gs_metric = self.config_dict['gs_metric']
            gs_cv = self.config_dict['gs_cv']
            gs_nfold = self.config_dict['gs_nfold']
            gs_verbose_eval = self.config_dict['gs_verbose_eval']
            gs_do_preds = self.config_dict['gs_do_preds']
            
            X_train = train[feature_cols]
            y_train = train[label_col]
            X_val = val[feature_cols]
            y_val = val[label_col]
            X_test = test[feature_cols]
            
            self._lgb_grid_search(X_train, y_train, X_val, y_val,
                                 categorical_features, search_rounds=gs_search_rounds, 
                                 filename_for_gs_results=gs_record, metric=gs_metric,
                                 cv=gs_cv, nfold=gs_nfold, verbose_eval=100,
                                 do_preds=gs_do_preds, X_test=X_test, preds_save_path=self.OUTDIR+'gs_saved_preds/')
        #self._renew_status(dictionary, 'grid_search', self.LOGDIR + 'todo_list.json')
    
    def _lgb_grid_search(self, X_train, y_train, X_val, y_val, categorical_feature, search_rounds, 
                        filename_for_gs_results, metric, cv, nfold, 
                        verbose_eval, do_preds, X_test, preds_save_path):
        import time
        import lightgbm as lgb
        
        if do_preds:
            if not os.path.exists(preds_save_path):
                os.makedirs(preds_save_path)

        for i in range(search_rounds):
            try:
                seed = int(time.time()* 1000000) % 45234634
                np.random.seed(seed)
                time.sleep(1) # sleep 1 sec to make sure the run_id is unique
                run_id = int(time.time()) # also works as the index of the result dataframe

                lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)

                lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=categorical_feature)

                lgb_params = {
                    'objective': 'binary',
                    'boosting': 'gbdt',
                    'num_rounds': 40,
                    'learning_rate': np.random.choice([0.1,0.03,0.001]),
                    'num_leaves': np.random.choice([15,31,61,127]),
                    'num_threads': 4, # best speed: set to number of real cpu cores, which is vCPU/2
                    'max_depth': -1, # no limit. This is used to deal with over-fitting when #data is small.
                    'min_data_in_leaf': np.random.randint(20,50),  #minimal number of data in one leaf. 
                    'feature_fraction': np.random.randint(3,11)/10,
                    'feature_fraction_seed': seed,
                    'early_stopping_round':1,
                    'bagging_freq': 1, #0 means disable bagging. k: perform bagging at every k iteration
                    'bagging_fraction': np.random.randint(3,11)/10, #Randomly select part of data 
                    'bagging_seed': seed,
                    'scale_pos_weight': 2,
                    'metric' : metric
                }
    #             import pprint
    #             pp = pprint.PrettyPrinter(indent=4)
    #             pp.pprint(lgb_params)

                lgb_params['timestamp'] = utils.get_time()
                if cv:
                    eval_hist = lgb.cv(lgb_params, lgb_train, nfold=nfold, 
                                       categorical_feature=categorical_feature, verbose_eval=verbose_eval, seed=seed)
                    best_rounds = len(eval_hist[metric+'-mean']) 
                    lgb_params['best_round'] = best_rounds
                    lgb_params['val_'+metric] = eval_hist[metric+'-mean'][-1]
                    lgb_params['cv'] = True
                else:
                    model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_val], verbose_eval=verbose_eval)
                    lgb_params['best_round'] = model.best_iteration
                    lgb_params['val_'+metric] = model.best_score['valid_1'][metric]
                    lgb_params['train_'+metric] = model.best_score['training'][metric]


                if do_preds:
                    self.logger.debug('[do_preds] is True, generating predictions ...')
                    if cv:
                        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
                        model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_train, 
                                          num_boost_round=best_rounds, verbose_eval=int(0.2*best_rounds))
                    y_test = model.predict(X_test)
                    np.save(preds_save_path+'preds_{}'.format(run_id),  y_test)
                    lgb_params['preds'] = run_id 
                    self.logger.debug('Predictions({}) saved in {}.'.format(run_id, preds_save_path))


                for k, v in lgb_params.items():
                    if isinstance(v, list):
                        lgb_params[k] = '"'+str(v)+'"'
                        self.logger.debug(lgb_params[k])

                res = pd.DataFrame(lgb_params, index=[run_id])

                if not os.path.exists(filename_for_gs_results):
                    res.to_csv(filename_for_gs_results)
                    self.logger.debug(filename_for_gs_results + ' created')
                else:
                    old_res = pd.read_csv(filename_for_gs_results, index_col='Unnamed: 0')
                    res = pd.concat([old_res, res])
                    res.to_csv(filename_for_gs_results)
                    self.logger.debug(filename_for_gs_results + ' updated')

            except Exception as e:
                if 'ResourceExhaustedError' in str(type(e)): # can't catch this error directly... 
                    self.logger.warning('Oops! ResourceExhaustedError. Continue next round')
                    continue
                else:
                    self.logger.error(e)
                    break
                
    ######### support functions #########
    # create a feature
    """ 
    feature_engineering todo list
    feature_engineering.txt line: <function_name>__<feature_combination_name>__<possible_param>
    file_name: train__<function_name>__<feature_combination_name>__<possible_param>.pkl
                test__<function_name>__<feature_combination_name>__<possible_param>.pkl
    """
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
        #feature_cols = '_'.join(feature_to_gen.get('feature_cols'))
        feature_cols = feature_to_gen.get('feature_cols')
        params = feature_to_gen.get('params')
        generated_feature_name = '__'.join([func.__name__, '_'.join(feature_cols)])
        if feature_to_gen.get('params') != {}: generated_feature_name += '__' + '_'.join(map(str, params.values()))
        params['train_len'] = self.train_len
        if not os.path.exists(self.FEATUREDIR + generated_feature_name + '.pkl'):
            #TODO: test if passing df=df[feature_cols+[self.label]] can save memory
            _df = func(df=self.df, cols=feature_to_gen.get('feature_cols'), dummy_col=self.label,
                              generated_feature_name=generated_feature_name, params=params)
            utils.save(df=_df, train_len=self.train_len, url=self.FEATUREDIR, name=generated_feature_name)
    
    # concat test
    def _concat_test(self, dictionary):
        if not dictionary['concat_test']:
            self._get_file_concat(base_df=self.test.copy(), 
                                  split_folder=self.FEATUREDIR, 
                                  concat_folder=self.DATADIR, 
                                  is_train=False, 
                                  file_name_body='test')
            self._renew_status(dictionary, 'concat_test', self.LOGDIR + 'todo_list.json')
        gc.collect()
    
    def _generate_down_sampling_index_file(self, has_file_built):
        
        def _downsampling(positive_idx, negative_idx, ratio):
            idx = np.random.choice(negative_idx, int(ratio*len(negative_idx)), replace=False)
            idx = np.concatenate((idx, positive_idx))
            return np.sort(idx).astype(int).tolist()
        
        index = []
        if has_file_built:
            with open(self.LOGDIR + 'down_sampling_idx.json', 'r') as file:
                index = json.load(file)
        else:
            train_exclude_val = self.train.drop(self.validation_index, axis=0)
            positive = list(train_exclude_val[train_exclude_val[self.label]==1].index.values)
            negative = list(train_exclude_val[train_exclude_val[self.label]==0].index.values)
            ratio = len(positive) / len(negative) if self.down_sampling_ratio == None else self.down_sampling_ratio 
            for i in range(self.down_sampling_amt): index.append(_downsampling(positive, negative, ratio))
            del train_exclude_val
            gc.collect()
            with open(self.LOGDIR + 'down_sampling_idx.json', 'w') as file:
                json.dump(index, file, indent=4, sort_keys=True)
        return index