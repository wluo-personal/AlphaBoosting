import os,sys
import numpy as np
import pandas as pd
# from tqdm import tqdm_notebook as tqdm
from sklearn.externals import joblib
sys.path.append(os.path.join(os.path.dirname(__file__),  '../LIB/'))
from env import ENV
from sklearn.preprocessing import normalize
# from tqdm import tqdm
import pickle
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostRegressor
import gc
import logging
logger = logging.getLogger('catBoostFILLNA')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def scan_nan_portion(df):
    portions = []
    columns = []
    for col in df.columns:
        columns.append(col)
        portions.append(np.sum(df[col].isnull())/len(df))
    return pd.Series(data=portions, index=columns)

def fill_na_list(X,th=200):
    re = scan_nan_portion(X)
#     re = re[re<0.15]
    re = re.sort_values()
    col_list = []
    na_list = []
    for col in re.index:
        if re[col] > 0 and X[col].nunique() >= th and str(X[col].dtypes)!= 'category' and str(X[col].dtypes)!= 'object':
            col_list.append(col)
            na_list.append(re[col])
    return col_list,na_list
            
def check_category_col(X,th=200):
    cate = []
    cols = list(X.columns)
    for col in cols:
        if X[col].nunique() < th or str(X[col].dtypes)== 'category' or str(X[col].dtypes)== 'object':
            cate.append(cols)
    return cate


def cate_index(df,category):
    index_list = []
    for col in df.columns:
        if col in category:
            index_list.append(category.index(col))
    return index_list
    
def trainCAT(na_train_x,na_val_x,na_train_y,na_val_y):
    cat_col = cate_index(na_train_x,categorys)
    model = CatBoostRegressor(iterations=4500, 
                              depth=None, 
                              thread_count=10,
                              learning_rate=0.01, 
                              loss_function='RMSE',
                              verbose=500,
                              task_type='GPU')
    model.fit(na_train_x,na_train_y,
              cat_features=cat_col,
              eval_set=[(na_val_x, na_val_y)],
              early_stopping_rounds=3,
              verbose_eval=500,
              metric_period=500)

#     reg.fit(na_train_x, na_train_y, eval_set=[(na_train_x,na_train_y),(na_val_x, na_val_y)],  verbose=200,early_stopping_rounds=100,eval_metric='l1' )
    return model

def saving(df_save):
    logger.info('saving...')
    train_save = df_save.iloc[:307511].copy()
    train_save['TARGET'] = targets
    logger.info(train_save.shape)

    test_save = df_save.iloc[307511:].copy()
    train_save.to_pickle(ENV.lgb_train_0827_na_extrem.value)
    test_save.to_pickle(ENV.lgb_test_0827_na_extrem.value)
    
    


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))



if __name__ == "__main__":
    train = pd.read_pickle(ENV.lgb_train_0827.value)

    logger.info('train shape is: {}'.format(train.shape))
    test = pd.read_pickle(ENV.lgb_test_0827.value)

    logger.info('test shape is: {}'.format(test.shape))
    try:
        train_fill = pd.read_pickle(ENV.lgb_train_0827_na_extrem.value)
        test_fill = pd.read_pickle(ENV.lgb_test_0827_na_extrem.value)
        filled_col = list(train_fill.columns)
        X_fill = pd.concat([train_fill,test_fill])
    except:
        train_fill = pd.DataFrame()
        test_fill = pd.DataFrame()
        X_fill = pd.DataFrame()
        filled_col = []
        logger.info('no filled NA found. Fresh start!')
    
    train['SK_ID_CURR'] = train['SK_ID_CURR'].astype(int)
    test['SK_ID_CURR'] = test['SK_ID_CURR'].astype(int)
    targets = train.TARGET.values

    train_id = train['SK_ID_CURR']
    test_id = test['SK_ID_CURR']

    X = pd.concat([train.drop('TARGET',axis=1),test])
    
    col_list,na_list = fill_na_list(X)
    categorys = check_category_col(X)
    
    
    
    Processed_list = []
    ignore_col = ['SK_ID_CURR']
    feature_col_base = list(set(X.columns) - set(ignore_col))
    val = 0.1
    save_count = 1
    count = 0

    ratio_file = os.path.join(os.path.dirname(__file__),  'ratio_stand.csv')

    for col,nav in zip(col_list,na_list):
        if col in filled_col:
            continue
        
        try:
            saved_df = pd.read_csv(ratio_file)
        except:
            logger.info('ratio file does not exist!')
            saved_df = pd.DataFrame()
        logger.info('start processing {} \nthe na is:{}'.format(col,nav))
        X[col] = X[col].replace(365243.0,0)
        X[col] = X[col].replace(np.NINF,np.NAN)
        X[col] = X[col].replace(np.PINF,np.NAN)
        X[col] = X[col].replace(np.Inf,np.NAN)
    #     feature_col = list(set(feature_col_base + Processed_list))
        feature_col = list(set(feature_col_base) - set([col]))

        na_train = X[X[col].notnull()][feature_col].copy()
        na_target = X[X[col].notnull()][col].copy()
        should_restar = True
        while should_restar:
            seed = np.random.randint(200) + 1 
            na_train_x,na_val_x,na_train_y,na_val_y = train_test_split(na_train,na_target, test_size=val,random_state=seed)
            logger.info(na_train.shape)
            na_test = X[X[col].isnull()][feature_col].copy()
            logger.info(na_test.shape)
            try:
                reg = trainCAT(na_train_x,na_val_x,na_train_y,na_val_y)
                should_restar = False
                X.loc[X[col].isnull(),col] = reg.predict(na_test)
            except:
                logger.error('catboost col has all NAN resplit')
                should_restar = True
        X_fill[col] = X[col].copy()
        Processed_list.append(col)
        logger.info('processed: {}/{}'.format(len(Processed_list),len(col_list)))
        count +=1
        if count % save_count == 0:
            saving(X_fill)
        logger.info('####################')
        meanNA = np.mean(na_train_y) * np.ones(len(na_val_y))
        preds = reg.predict(na_val_x)
        me_rmse = rmse(meanNA,na_val_y)
        es_rmse = rmse(preds,na_val_y)
        corr = np.corrcoef(preds,na_val_y).flatten()[1]
        logger.info('mean encoding: {}'.format(me_rmse))
        logger.info('estimate encoding: {}'.format(es_rmse))
        logger.info('me/estimate ratio is : {}'.format(me_rmse/es_rmse))
        logger.info('correlation in val: {}'.format(corr))
        new_df = pd.DataFrame({'meanEncoRMSE':[me_rmse],
                               'catBoostEncoRMSE':[es_rmse],
                               'meByCatRatio':[me_rmse/es_rmse],
                               'colName':[col],
                               'corr':[corr],
                               'preds_mean':[np.mean(preds)],
                               'preds_std':[np.std(preds)],
                               'val_mean':[np.mean(na_val_y)],
                               'val_std':[np.std(na_val_y)],
                               'naRatio':[nav]})
        conc = pd.concat([saved_df,new_df])
        conc.to_csv(ratio_file,index=False)
        print('####################')
        del reg
        gc.collect()
