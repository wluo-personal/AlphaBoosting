import os,sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'../LIB/'))
import logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - CLASS:%(name)s- METHOD:%(funcName)s -LINE:%(lineno)d - MSG:%(message)s')
sh.setFormatter(sh_formatter)
module_logger.addHandler(sh)




from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU,CuDNNGRU,Flatten,BatchNormalization,CuDNNLSTM,Activation,BatchNormalization
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.preprocessing import text, sequence
from sklearn.metrics import log_loss,roc_auc_score
from keras.regularizers import l1
from keras.regularizers import l2
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import re
from keras import backend as K
import numpy as np
import gc
import time








def train_each_fold(get_nn_model,input_train_dict,input_val_dict,y_train,y_val,cols,doc_col=[],tolerance=30,train_batch_size=5000):
    model = get_nn_model(cols,doc_col)
    cur_to = 0
    best_logloss = None
    best_weights = None
    count = 0
    base_lr = 0.001
    while True:
        model.fit(input_train_dict, y_train, 
                  batch_size=train_batch_size, 
                  epochs=1,
                  verbose=2,
                  shuffle=True,
                  )
        preds = model.predict(input_val_dict,25000,verbose=2)
        logloss = log_loss(y_val,preds)
        roc = roc_auc_score(y_val,preds)
        print(logloss)
        print(roc)
        if best_logloss is None:
            best_logloss = logloss
            best_weights = model.get_weights()
        else:
            if best_logloss > logloss:
                best_logloss = logloss
                best_weights = model.get_weights()
                cur_to = 0
            else:
                cur_to +=1
        if cur_to == tolerance:
            break
        module_logger.info('best logloss is: {}'.format(best_logloss))
        module_logger.info('remainning trial is: {}/{}'.format(cur_to,tolerance))
        module_logger.info('total epoch trained: {}'.format(count))
        count+=1
    model.set_weights(best_weights)
    return model





def nn_K_fold(get_nn_model,
              train_fold_dict,
              val_fold_dict,
              train_fold_y,
              val_fold_y,
              test_input_dict,
              val_index_list,
              train_df,
              test_df,
              pred_col_name = 'predicted_score',
              holdout_input_dict=None,
              holdout_y=None,
              holdout_index_list=None,
              nondoc_cols=[],
              doc_cols=[],
              tolerance=30,
              train_batch_size=5000,
              preds_batch=5000):
    """
    train_fold_dict: format, key - foldNum, value - nn input 
    train_fold_y: label for train fold, fotmat, key -foldNum, value - label
    val_fold_dict: format, key - foldNum, value - nn input 
    val_fold_y: label for val fold, fotmat, key -foldNum, value - label
    test_input_dict: format, key - foldNum, value - nn input 
    holdout_input_dict: Noneable, if none, not using holdout
    holdout_y: Noneable, label for holdout
    cols: all cols used to do ebd
    doc_cols, cols that are used to do rnn
    val_index_list: cv, going to predict and generate oof
    holdout_index_list: going to predict on each fold
    train_df: 'dataframe which only has id columns, which will be used to store oof prediction'
    test_df: 'dataframe which only has id columns, which will be used to store test prediction'
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    n_fold = len(train_fold_dict)
    test_preds_list = []
    val_score_list = []
    hold_out_preds_list = []
    holdout_score_list = []
    train_score_list = []
    cv_score_list = []
    train_df[pred_col_name] = np.nan
    test_df[pred_col_name] = np.nan
    
    for fold in range(n_fold):
        module_logger.info('start fold {}...'.format(fold))
        model = train_each_fold(get_nn_model,
                                train_fold_dict[fold],
                                val_fold_dict[fold],
                                train_fold_y[fold],
                                val_fold_y[fold],
                                nondoc_cols,
                                doc_cols,
                                tolerance=tolerance,
                                train_batch_size=train_batch_size)
        train_preds =  model.predict(train_fold_dict[fold],preds_batch,verbose=2)
        val_preds = model.predict(val_fold_dict[fold],preds_batch,verbose=2)
        train_loss = log_loss(train_fold_y[fold],train_preds)
        val_loss = log_loss(val_fold_y[fold],val_preds)
        train_df.loc[val_index_list[fold],pred_col_name] = val_preds
        test_preds = model.predict(test_input_dict,preds_batch,verbose=2)
        test_preds_list.append(test_preds)
        val_score_list.append(val_loss)
        train_score_list.append(train_loss)
        module_logger.info('Fold {} finish! val loss: {}.'.format(fold,val_loss))
        if holdout_index_list is not None:
            ho_preds = model.predict(holdout_input_dict,preds_batch,verbose=2)
            ho_loss = log_loss(holdout_y,ho_preds)
            ho_roc = roc_auc_score(holdout_y,ho_preds)
            holdout_score_list.append(ho_loss)
            hold_out_preds_list.append(ho_preds)
            module_logger.info('hold out loss: {}'.format(ho_loss))
        del model
        gc.collect()
        time.sleep(5)
            
    module_logger.info('finish training... calculating evl matrix')
    test_preds_list = np.array(test_preds_list)
    hold_out_preds_list = np.array(hold_out_preds_list)
    test_preds_final = np.mean(test_preds_list,axis=0)
    cv_score_mean = np.mean(val_score_list)
    train_score_mean = np.mean(train_score_list)
    test_df[pred_col_name] = test_preds_final
    module_logger.info('cv mean is: {}'.format(cv_score_mean))
    if holdout_index_list is not None:
        ho_preds_final = np.mean(hold_out_preds_list,axis=0)
        ho_loss_overall = log_loss(holdout_y,ho_preds_final)
        ho_roc_overall = roc_auc_score(holdout_y,ho_preds_final)
        train_df.loc[holdout_index_list,pred_col_name] = ho_preds_final
        module_logger.info('holdout loss overall is: {}'.format(ho_loss_overall))
        module_logger.info('holdout roc overall is: {}'.format(ho_roc_overall))
        return train_df,test_df,cv_score_mean,ho_loss_overall,train_score_mean
    else:
        return train_df,test_df,cv_score_mean,0.0,train_score_mean