import time, os
from datetime import timedelta
import lightgbm as lgb
import pandas as pd
import numpy as np
from automl_libs import utils
import logging
module_logger = logging.getLogger(__name__)

def lgb_grid_search(X_train, y_train, X_val, y_val, categorical_feature, search_rounds, 
                        filename_for_gs_results, gs_params_gen, cv, nfold, 
                        verbose_eval, do_preds, X_test, preds_save_path, suppress_warning):
    
    if suppress_warning:
        import warnings
        warnings.filterwarnings("ignore")

    if do_preds:
        if not os.path.exists(preds_save_path):
            os.makedirs(preds_save_path)

    for i in range(search_rounds):
        try:
            lgb_params, seed = gs_params_gen()
            metric = lgb_params['metric']
            time.sleep(1) # sleep 1 sec to make sure the run_id is unique
            run_id = int(time.time()) # also works as the index of the result dataframe

            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)

            lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=categorical_feature)

#             import pprint
#             pp = pprint.PrettyPrinter(indent=4)
#             pp.pprint(lgb_params)
            lgb_params['timestamp'] = utils.get_time()
            gs_start_time = time.time()
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
                                      num_boost_round=best_rounds, verbose_eval=int(0.2*best_rounds))
                module_logger.debug('Predicting...')
                y_test = model.predict(X_test)
                np.save(preds_save_path+'preds_{}'.format(run_id),  y_test)
                predict_elapsed_time_as_hhmmss = str(timedelta(seconds=int(time.time() - predict_start_time)))
                lgb_params['pred_timespent'] = predict_elapsed_time_as_hhmmss
                module_logger.debug('Predictions({}) saved in {}.'.format(run_id, preds_save_path))

            # remove params not needed to be recorded in grid search history csv
            lgb_params.pop('categorical_column',None)
            lgb_params.pop('verbose',None)
                
            # so that [1,2,3] can be converted to "[1,2,3]" and be treated as a whole in csv
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