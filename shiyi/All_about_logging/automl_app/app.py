import os, json, gc
import pandas as pd
import numpy as np
from automl_libs import Utils

class AlphaBoosting:
    def __init__(self, root=None, train_csv_url=None, test_csv_url=None, validation_index=None, func_map=None, timestamp=None,
                 label=None, categorical_features=None, numerical_features=None, validation_ratio=0.1, ngram=(1,1),
                 downsampling=1, down_sampling_ratio=None, run_record='run_record.json'):
        downsampling_amount_changed = False
        down_sampling_ratio_changed = False
        val_index_changed = False
        if run_record == None:
            raise Exception('run record file can not be None')
        
        self.func_map = func_map
        if not os.path.exists(run_record):
            # set run record 
            print('First time running...')
            self.ROOT = root
            self.OUTDIR = root + 'output/'
            self.LOGDIR = root + 'log/'
            self.DATADIR = root + 'data/'
            self.train_csv_url = train_csv_url
            self.test_csv_url = test_csv_url
            self.timestamp = timestamp
            self.label = label
            self.categorical_features = categorical_features
            self.numerical_features = numerical_features
            self.downsampling = downsampling
            self.down_sampling_ratio = down_sampling_ratio 
            # read data
            self._read_data()
            if validation_index == None:
                self.validation_index = list(range(int(self.train_len*(1-validation_ratio)), self.train_len))
            else:
                self.validation_index = validation_index
        else:
            print('Continue from the previous run...')
            with open(run_record, 'r') as infile: file = json.load(infile)
            self.ROOT = file['root']
            self.OUTDIR = file['root'] + 'output/'
            self.LOGDIR = file['root'] + 'log/'
            self.DATADIR = file['root'] + 'data/'
            self.train_csv_url = file['train_csv_url']
            self.test_csv_url = file['test_csv_url']
            # read data
            self._read_data()
            self.timestamp = file['timestamp']
            self.label = file['label']
            self.categorical_features = file['categorical_features']
            self.numerical_features = file['numerical_features']
            self.validation_index = json.load(open(file['validation_index']))
            self.downsampling = file['downsampling']
            self.down_sampling_ratio = file['down_sampling_ratio'] 
            
            # check if validation and down sampling need to be redone
            old_down_sampling = self.downsampling
            if downsampling != self.downsampling:
                downsampling_amount_changed = True
                self.downsampling = downsampling
            if down_sampling_ratio != None and self.down_sampling_ratio != down_sampling_ratio:
                down_sampling_ratio_changed = True
                self.down_sampling_ratio = down_sampling_ratio
            if validation_index != None and self.validation_index != validation_index:
                val_index_changed = True
                self.validation_index = validation_index
        
        # build relavent directories
        self.FEATUREDIR = self.DATADIR + 'features/'
        if not os.path.exists(self.OUTDIR): os.makedirs(self.OUTDIR)
        if not os.path.exists(self.LOGDIR): os.makedirs(self.LOGDIR)
        if not os.path.exists(self.DATADIR): os.makedirs(self.DATADIR)
        if not os.path.exists(self.FEATUREDIR): os.makedirs(self.FEATUREDIR)
            
        # save run_record:
        print('save run record')
        self._save_run_record(run_record)
        
        # generate todo list: c
        print('generate todo list')
        dictionary = self._generate_todo_list()
        
        if downsampling_amount_changed or down_sampling_ratio_changed or val_index_changed:
            dictionary['val_downsample_generate_index'] = False
            dictionary['val_downsample_split'] = False
            dictionary['val_downsample_generation'] = False
            
            if os.path.exists(self.LOGDIR + 'down_sampling_idx.json'): os.remove(self.LOGDIR + 'down_sampling_idx.json')
            if os.path.exists(self.LOGDIR + 'val.pkl'): os.remove(self.DATADIR + 'val.pkl')
            for i in range(old_down_sampling): 
                if os.path.exists(self.DATADIR + str(i) + '.pkl'):
                    os.remove(self.DATADIR + str(i) + '.pkl')
            shutil.rmtree(self.DATADIR + 'split/')
        
        # feature engineering
        print('feature engineering')
        self._feature_engineering(dictionary)
        
        # get validation
        print('validation')
        self._validation_and_down_sampling(dictionary)
        
        # concatenant test: c
        print('concat test')
        self._concat_test(dictionary)
        
        # grid search
        print('grid search')
        self._grid_search(dictionary)
    
    
    ######### util functions #########
    def _read_data(self):
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        if self.train_csv_url != None: self.train = pd.read_csv(self.train_csv_url)
        if self.test_csv_url != None: self.test = pd.read_csv(self.test_csv_url)
        self.df = pd.concat([self.train, self.test], ignore_index=True)
        self.train_len = self.train.shape[0]
        
    def _renew_status(self, dictionary, key, file_url):
        dictionary[key] = True
        with open(file_url, 'w') as f:
            json.dump(dictionary, f, indent=4, sort_keys=True)

            
    def _save_run_record(self, url):
        val_index_url = self.LOGDIR + 'val_index.json'
        d = {
            'root':                 self.ROOT,
            'train_csv_url':        self.train_csv_url,
            'test_csv_url':         self.test_csv_url,
            'timestamp':            self.timestamp,
            'label':                self.label,
            'categorical_features': self.categorical_features,
            'numerical_features':   self.numerical_features,
            'validation_index':     val_index_url, 
            'downsampling':         self.downsampling,
            'down_sampling_ratio':  self.down_sampling_ratio
        }
        with open(val_index_url, 'w') as f: json.dump(self.validation_index, f, indent=4, sort_keys=True)
        with open(url, 'w') as f: json.dump(d, f, indent=4, sort_keys=True)
        del d
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
            dictionary = {'feature_engineering':           False, 
                          'val_downsample_generate_index': self.downsampling==0,
                          'val_downsample_split':          self.downsampling==0,
                          'val_downsample_generation':     False,
                          'concat_test':                   False,
                          'grid_search':                   False}
            with open(self.LOGDIR + 'todo_list.json', 'w') as file: 
                json.dump(dictionary, file, indent=4, sort_keys=True)
        return dictionary
    
    def _feature_engineering(self, dictionary):
        feature_engineering_file_url = self.LOGDIR + 'feature_engineering.json'
        if not dictionary['feature_engineering']:
            if not os.path.exists(feature_engineering_file_url):
                self._generate_feature_engineering_file(feature_engineering_file_url)
            with open(feature_engineering_file_url, 'r') as file:
                data = json.load(file)
                for line in data:
                    self._add_column(line, self.func_map)
        self._renew_status(dictionary, 'feature_engineering', (self.LOGDIR + 'todo_list.json'))
    
    def _validation_and_down_sampling(self, dictionary):
        split_folder = []
        index = []
        down_sampling_url = None
        if not dictionary['val_downsample_generation']:
            # down sampling
            down_sampling_url = self.DATADIR + 'split/'
            if not os.path.exists(down_sampling_url): os.makedirs(down_sampling_url)
            index.extend(self._generate_down_sampling_index_file(dictionary['val_downsample_generate_index']))
            for i in range(self.downsampling): 
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
            if self.downsampling == 0:
                index.append(sorted(list(set(range(self.train_len)).difference(set(self.validation_index)))))
                split_folder.append(self.FEATUREDIR)
            for i in range(len(split_folder)):
                file_name_body = 'val' if split_folder[i] == self.FEATUREDIR else split_folder[i].split('/')[-2]
                self._get_file_concat(base_df=self.train.loc[index[i]].copy().reset_index(drop=True),
                                      split_folder=split_folder[i], 
                                      concat_folder=self.DATADIR, 
                                      is_train=True, 
                                      file_name_body=file_name_body)
            self._renew_status(dictionary, 'val_downsample_generation', self.LOGDIR + 'todo_list.json')
        
    
    def _grid_search(self, dictionary):
        if not dictionary['grid_search']:
            train = pd.read_pickle('data/0.pkl')
            val = pd.read_pickle('data/val.pkl')
            test = pd.read_pickle('data/test.pkl')
#             feature_cols = train.columns - label - not_feature_cols
#             label =
#             X_train = train[feature_cols]
#             y_train = train[label]
#             self._lgb_grid_search()
        self._renew_status(dictionary, 'grid_search', self.LOGDIR + 'todo_list.json')
    
    def _lgb_grid_search(X_train, y_train, X_val, y_val, categorical_feature, search_rounds, 
                        filename_for_gs_results, metric='auc', cv=False, nfold=5, 
                        verbose_eval=50, X_test=None, preds_save_path=None):
        import time
        import lightgbm as lgb
        
        def get_time(timezone='America/New_York', time_format='%Y-%m-%d %H:%M:%S'):
            from datetime import datetime
            from dateutil import tz

            # METHOD 1: Hardcode zones:
            from_zone = tz.gettz('UTC')
            to_zone = tz.gettz(timezone)

            utc = datetime.utcnow()

            # Tell the datetime object that it's in UTC time zone since 
            # datetime objects are 'naive' by default
            utc = utc.replace(tzinfo=from_zone)

            # Convert time zone
            est = utc.astimezone(to_zone)

            return est.strftime(time_format)
    
        if X_test is not None:
            if preds_save_path is None:
                preds_save_path = 'LGB_GS_SAVES/'
                if not os.path.exists(preds_save_path):
                    os.makedirs(preds_save_path)
                print('No save path provides, {} will be used to save predictions'.format(preds_save_path))
            else:
                if not os.path.exists(preds_save_path):
                    raise ValueError('{} path does not exist. Mission aborted.'.format())

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
                    'num_rounds': 4000,
                    'learning_rate': np.random.choice([0.1,0.03,0.001]),
                    'num_leaves': np.random.choice([15,31,61,127]),
                    'num_threads': 4, # best speed: set to number of real cpu cores, which is vCPU/2
                    'max_depth': -1, # no limit. This is used to deal with over-fitting when #data is small.
                    'min_data_in_leaf': np.random.randint(20,50),  #minimal number of data in one leaf. 
                    'feature_fraction': np.random.randint(3,11)/10,
                    'feature_fraction_seed': seed,
                    'early_stopping_round':70,
                    'bagging_freq': 1, #0 means disable bagging. k: perform bagging at every k iteration
                    'bagging_fraction': np.random.randint(3,11)/10, #Randomly select part of data 
                    'bagging_seed': seed,
                    'scale_pos_weight': 2,
                    'metric' : metric
                }
    #             import pprint
    #             pp = pprint.PrettyPrinter(indent=4)
    #             pp.pprint(lgb_params)

                lgb_params['timestamp'] = get_time()
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


                if X_test is not None:
                    print('X_test is not None, generating predictions ...')
                    if cv:
                        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
                        model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_train, 
                                          num_boost_round=best_rounds, verbose_eval=int(0.2*best_rounds))
                    y_test = model.predict(X_test)
                    np.save(preds_save_path+'preds_{}'.format(run_id),  y_test)
                    lgb_params['preds'] = run_id 
                    print('Predictions({}) saved in {}.'.format(run_id, preds_save_path))


                for k, v in lgb_params.items():
                    if isinstance(v, list):
                        lgb_params[k] = '"'+str(v)+'"'
                        print(lgb_params[k])

                res = pd.DataFrame(lgb_params, index=[run_id])

                if not os.path.exists(filename_for_gs_results):
                    res.to_csv(filename_for_gs_results)
                    print(filename_for_gs_results, 'created')
                else:
                    old_res = pd.read_csv(filename_for_gs_results, index_col='Unnamed: 0')
                    res = pd.concat([old_res, res])
                    res.to_csv(filename_for_gs_results)
                    print(filename_for_gs_results, 'updated')

            except Exception as e:
                if 'ResourceExhaustedError' in str(type(e)): # can't catch this error directly... 
                    print('Oops! ResourceExhaustedError. Continue next round')
                    continue
                else:
                    print(e)
                    break
                
    ######### support functions #########
    # feature engineering
    def _generate_feature_engineering_file(self, feature_engineering_file_url):
        with open(feature_engineering_file_url, 'w') as file:
            dictionary = []
            
            # params
            param = {'trainLen': self.train_len, 'splitCol': 'a', 'col': self.label, 'coefficient': 10, 'n': 2, 'fillna': 22}
            
            dictionary.append({'params': param, 'function': 'count', 'feature_cols': ['a','b']})
            dictionary.append({'params': param, 'function': 'unique_count', 'feature_cols': ['a','b']})
            dictionary.append({'params': param, 'function': 'cumulative_count', 'feature_cols': ['a','b']})
            dictionary.append({'params': param, 'function': 'reverse_cumulative_count', 'feature_cols': ['a','b']})
            dictionary.append({'params': param, 'function': 'variance', 'feature_cols': ['a','n']})
            dictionary.append({'params': param, 'function': 'count_std_over_mean', 'feature_cols': ['a','b']})
            dictionary.append({'params': param, 'function': 'time_to_n_next', 'feature_cols': ['a','t']})
            dictionary.append({'params': param, 'function': 'count_in_previous_n_time_unit', 'feature_cols': ['a','t']})
            dictionary.append({'params': param, 'function': 'count_in_next_n_time_unit', 'feature_cols': ['a','t']})
            dictionary.append({'params': param, 'function': 'woe', 'feature_cols': ['b']})
            dictionary.append({'params': param, 'function': 'chi_square', 'feature_cols': ['b']})
            dictionary.append({'params': param, 'function': 'mean', 'feature_cols': ['b']})
            
            json.dump(dictionary, file, indent=4, sort_keys=True)
    
    # create a feature
    """ 
    feature_engineering todo list
    feature_engineering.txt line: <function_name>__<feature_combination_name>__<possible_param>
    file_name: train__<function_name>__<feature_combination_name>__<possible_param>.pkl
                test__<function_name>__<feature_combination_name>__<possible_param>.pkl
    """
    def _add_column(self, line, f_map):
        fun = line.get('function')
        feature = '_'.join(line.get('feature_cols'))
        col_name = '__'.join([fun, feature])
        if line.get('params') != None: col_name += '__' + '_'.join(map(str, line.get('params').values()))
        if not os.path.exists(self.FEATUREDIR + col_name + '.pkl'):
            _df = f_map[fun](df=self.df, cols=line.get('feature_cols'), col_name=col_name, params=line.get('params'))
            Utils.save(df=_df, train_len=self.train_len, url=self.FEATUREDIR, name=col_name)
    
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
            for i in range(self.downsampling): index.append(_downsampling(positive, negative, ratio))
            del train_exclude_val
            gc.collect()
            with open(self.LOGDIR + 'down_sampling_idx.json', 'w') as file:
                json.dump(index, file, indent=4, sort_keys=True)
        return index