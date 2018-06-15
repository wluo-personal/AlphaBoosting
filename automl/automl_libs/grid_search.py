import time, os
import lightgbm as lgb
import pandas as pd
import numpy as np
from automl_libs import utils
import logging
module_logger = logging.getLogger(__name__)

def lgb_grid_search(X_train, y_train, X_val, y_val, categorical_feature, search_rounds, 
                        filename_for_gs_results, metric, cv, nfold, 
                        verbose_eval, do_preds, X_test, preds_save_path):

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

            import pdb
            pdb.set_trace()
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
                module_logger.debug('[do_preds] is True, generating predictions ...')
                if cv:
                    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
                    model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_train, 
                                      num_boost_round=best_rounds, verbose_eval=int(0.2*best_rounds))
                y_test = model.predict(X_test)
                np.save(preds_save_path+'preds_{}'.format(run_id),  y_test)
                lgb_params['preds'] = run_id 
                module_logger.debug('Predictions({}) saved in {}.'.format(run_id, preds_save_path))


            for k, v in lgb_params.items():
                if isinstance(v, list):
                    lgb_params[k] = '"'+str(v)+'"'
                    #module_logger.debug(lgb_params[k])

            res = pd.DataFrame(lgb_params, index=[run_id])

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
                module_logger.error(e)
                break