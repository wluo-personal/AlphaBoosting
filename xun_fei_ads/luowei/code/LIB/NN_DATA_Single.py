import os,sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'../LIB/'))
from env import FILE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from tqdm import tqdm
import time
import numpy as np
import pickle
from NN_K_FOLD import *
import numpy as np

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

import logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - CLASS:%(name)s- METHOD:%(funcName)s -LINE:%(lineno)d - MSG:%(message)s')
sh.setFormatter(sh_formatter)
module_logger.addHandler(sh)



def save_oof(train_df,test_df,cv,ta,ho=None,file_name='',path='../../data/nn_ebd/'):
    try:
        report = pd.read_csv(path+'report.csv')
    except:
        print('no report found! generate a new one!')
        report = pd.DataFrame()
    new_record = pd.DataFrame({'ho':[ho],'cv':[cv],'train_mean':[ta],'file':[file_name]})
    report = pd.concat([report,new_record],sort=False)
    train_df.to_pickle(path+'train/'+file_name+'.pkl')
    print(train_df.shape)
    test_df.to_csv(path+'test/'+file_name+'.csv',index=False)
    print(test_df.shape)
    report.to_csv(path+'report.csv',index=False)
    print('done!')


        
def get_tok(X,col,train_length,d_filter='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    X[col] = X[col].astype(str)

    tok_all=text.Tokenizer(num_words=X[col].nunique(),lower=False,filters=d_filter)
    tok_all.fit_on_texts(list(X[col].values))

    tok_train=text.Tokenizer(num_words=X[col].iloc[:train_length].nunique(),lower=False,filters=d_filter)
    tok_train.fit_on_texts(list(X[col].iloc[:train_length].values))

    tok_test=text.Tokenizer(num_words=X[col].iloc[train_length:].nunique(),lower=False,filters=d_filter)
    tok_test.fit_on_texts(list(X[col].iloc[train_length:].values))
    word_intersection = set(tok_train.word_index.keys()).intersection(set(tok_test.word_index.keys()))

    self_index = {}
    count = 1
    for word in word_intersection:
        self_index[word] = count
        count+=1
    self_index['unknown'] = count
    print('max index is: {}'.format(count))

    for word in tok_all.word_index.keys():
        tok_all.word_index[word] = self_index.get(word,count)
    return tok_all
        
def encap_data_dict(X,train_length,
                    train_index,
                    holdout_index=None,
                    target='click',
                    ignore_columns=[],
                    doc_col=['user_tags','model'],
                    non_doc_col_append=[]):
    
    need_process_col = list(set(X.columns) - set(ignore_columns))
    X_ = X[need_process_col].copy()
    non_doc_col = [f for f in need_process_col if f not in doc_col]
    non_doc_col += non_doc_col_append
    X_doc = X[doc_col].copy()
    for col in tqdm(non_doc_col):
        test_values = set(X_[col].iloc[train_length:].astype(str).unique())
        train_values = set(X_[col].iloc[:train_length].astype(str).unique())
        intersection = train_values.intersection(test_values)
        out_liyer = list(train_values.union(test_values) - intersection)
        if len(out_liyer) > 0:
            out_liyer_mapping = pd.Series(index=list(out_liyer),data=1)
            filtered = (X_[col].astype(str).map(out_liyer_mapping) == 1)
            X_.loc[filtered,col] = np.nan
        X_[col] = le.fit_transform(X_[col].astype(str))
        X_[col] = col + '_'+X_[col].astype(str)
    
    for col in tqdm(doc_col):
        X_doc[col] = X_doc[col].astype(str)
    
    
    
    
 
    train = X.iloc[:train_length].copy()
    train_index_list = []
    val_index_list = []
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    for t,v in folds.split(train.loc[train_index],train.loc[train_index,target]):
        train_index_list.append(train.loc[train_index].iloc[t].index)
        val_index_list.append(train.loc[train_index].iloc[v].index)
    train_fold_y = {}
    val_fold_y = {}
    if holdout_index is not None:
        holdout_y = train.loc[holdout_index,'click'].values
    else:
        holdout_y = None
    for fold in range(num_folds):
        train_fold_y[fold] = train.loc[train_index_list[fold],'click'].values
        val_fold_y[fold] = train.loc[val_index_list[fold],'click'].values      
    #################################################################################################################
    info_dict = {}
    train_all_dict = {}
    train_fold_dict = {}
    val_fold_dict = {}
    holdout_input_dict = {}
    test_input_dict = {}
    maxlen = 1
    # process non doc col
    for col in tqdm(non_doc_col):
        maxlen = 1
        tok=text.Tokenizer(num_words=X_[col].nunique(),lower=False,filters='@')
        tok.fit_on_texts(list(X_[col]))
        info_dict.update({prefix_input_nonDoc+col:{'tok':tok}})
        t = tok.texts_to_sequences(list(X_[col].iloc[:train_length].values))
        te = tok.texts_to_sequences(list(X_[col].iloc[train_length:].values))
        train_all_dict[prefix_input_nonDoc+col] = sequence.pad_sequences(t,maxlen=maxlen)
        test_input_dict[prefix_input_nonDoc+col] = sequence.pad_sequences(te,maxlen=maxlen)
        if holdout_index is not None:
            holdout_input_dict[prefix_input_nonDoc+col] = train_all_dict[prefix_input_nonDoc+col][list(holdout_index)]

        for fold in range(num_folds):
            if train_fold_dict.get(fold) is None:
                train_fold_dict[fold] = {}
                val_fold_dict[fold] = {}
            train_fold_dict[fold].update({prefix_input_nonDoc+col: train_all_dict[prefix_input_nonDoc+col][list(train_index_list[fold])]})
            val_fold_dict[fold].update({prefix_input_nonDoc+col:train_all_dict[prefix_input_nonDoc+col][list(val_index_list[fold])]})
        
    sequence_size_dict = {}
    for col in tqdm(doc_col):
        if col == 'user_tags':
            maxlen = 50
            tok = get_tok(X_doc,col=col,train_length=train_length,d_filter=',')
#             tok=text.Tokenizer(num_words=X_doc[col].nunique(),lower=False,filters=',')
#             tok.fit_on_texts(list(X_doc[col]))

        elif col == 'model':
            maxlen = 15
            tok=text.Tokenizer(num_words=X_doc[col].nunique(),lower=False)
            tok.fit_on_texts(list(X_doc[col]))
        else:
            maxlen = 15
            tok=text.Tokenizer(num_words=X_doc[col].nunique(),lower=False)
            tok.fit_on_texts(list(X_doc[col]))
        info_dict.update({prefix_input_Doc+col:{'tok':tok}})
        sequence_size_dict[col] = maxlen
        t = tok.texts_to_sequences(list(X_doc[col].iloc[:train_length].values))
        te = tok.texts_to_sequences(list(X_doc[col].iloc[train_length:].values))
        train_all_dict[prefix_input_Doc+col] = sequence.pad_sequences(t,maxlen=maxlen)
        test_input_dict[prefix_input_Doc+col] = sequence.pad_sequences(te,maxlen=maxlen)
        if holdout_index is not None:
            holdout_input_dict[prefix_input_Doc+col] = train_all_dict[prefix_input_Doc+col][list(holdout_index)]

        for fold in range(num_folds):
            if train_fold_dict.get(fold) is None:
                train_fold_dict[fold] = {}
                val_fold_dict[fold] = {}
            train_fold_dict[fold].update({prefix_input_Doc+col: train_all_dict[prefix_input_Doc+col][list(train_index_list[fold])]})
            val_fold_dict[fold].update({prefix_input_Doc+col:train_all_dict[prefix_input_Doc+col][list(val_index_list[fold])]})
    return info_dict,train_fold_dict,val_fold_dict,holdout_input_dict,test_input_dict,train_fold_y,val_fold_y,holdout_y,val_index_list,sequence_size_dict
        
    
    
    
def get_nn_model(cols,doc_cols=[],numu_cols=[]):
    """
    cols, used to do ebd and dense layers
    doc_cols: used to do rnn
    there can be overlaps
    """
    input_list = []
    concat_list = []
    numu_list = []
    for col in cols:
        max_feature = len(info_dict[prefix_input_nonDoc+col]['tok'].index_word)
        embed_size = int(np.log2(max_feature)/np.log2(1.5))
        if embed_size< 2:
            embed_size = 2
        cur_input = Input(shape=(1, ),name = prefix_input_nonDoc+col)
        
       
        embed_layer = Embedding(max_feature,
                            embed_size,
                            input_length=1,
                            trainable=True,
                            embeddings_regularizer=l2(0.0005),
                            name='ebd_'+col)(cur_input)
        embed_layer = SpatialDropout1D(0.5)(embed_layer)
        x = Flatten()(embed_layer)
        input_list.append(cur_input)
        concat_list.append(x)
    for col in doc_cols:
        max_feature = len(info_dict[prefix_input_Doc+col]['tok'].index_word)
        embed_size = int(np.log2(max_feature)/np.log2(1.5))
        if embed_size< 2:
            embed_size = 2
        input_shape = sequence_size_dict[col]
        cur_input = Input(shape=(input_shape, ),name = prefix_input_Doc+col)
        embed_layer = Embedding(max_feature,
                            embed_size,
                            input_length=input_shape,
                            trainable=True,
                            embeddings_regularizer=l2(0.0005),
                            name='ebd_rnn_'+col)(cur_input)
        x = SpatialDropout1D(0.5)(embed_layer)
        x = Bidirectional(CuDNNGRU(25, return_sequences=True))(x)
        x = Conv1D(25, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
        x_aveP = GlobalAveragePooling1D()(x)
        x_maxP = GlobalMaxPooling1D()(x)
        x = concatenate([x_aveP,x_maxP])
        input_list.append(cur_input)
        concat_list.append(x)

    if len(numu_cols) > 0:
        print('add numu...')
        nu_shape = len(numu_cols)
        cur_input = Input(shape=(nu_shape, ),name = prefix_input_nu)
        x = BatchNormalization()(cur_input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        input_list.append(cur_input)
        numu_list.append(x)
       
    if len(concat_list) > 1:
        x = concatenate(concat_list)
#     x = BatchNormalization()(x)
    

    if len(numu_list)>0:
        x = concatenate([x]+numu_list)
#         x = BatchNormalization()(x)
        
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    preds = Dense(1, activation="sigmoid")(x)
    model = Model(input_list, preds)
    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])    
    return model





if __name__ == '__main__':
    prefix_input_nonDoc = 'input_'
    prefix_input_Doc = 'input_rnn_'
    num_folds = 5
    seed = 1001
    
    train = pd.read_pickle(FILE.train_ori.value)
    module_logger.info('train shape is: {}'.format(train.shape))
    train_length = len(train)
    test = pd.read_pickle(FILE.test_ori.value)
    module_logger.info('test shape is: {}'.format(test.shape))
    
    train_index = train.index
    holdout_index = None
#     train_index = pickle.load(open(FILE.train_index.value,'rb'))
#     holdout_index = pickle.load(open(FILE.holdout_index.value,'rb'))

    X = pd.concat([train,test],sort=False)
    X_shiyi = pd.read_pickle(FILE.shiyi_fillna_ori.value)
    module_logger.info(X_shiyi.shape)

    X = X.merge(X_shiyi[['time_hour','instance_id']],how='inner',on='instance_id')


    ignore_columns = ['instance_id','time','click'] + ['creative_is_js', 'creative_is_voicead', 'app_paid']
    need_process_col = list(set(X.columns) - set(ignore_columns))
    X_ = X[need_process_col].copy()
    doc_col = ['user_tags','model']
    non_doc_col = [f for f in need_process_col if f not in doc_col]
    
    info_dict,train_fold_dict,val_fold_dict,holdout_input_dict,test_input_dict,train_fold_y,val_fold_y,holdout_y,val_index_list,sequence_size_dict =encap_data_dict(X,train_length,
                    train_index,
                    holdout_index,
                    target='click',
                    ignore_columns=ignore_columns,
                    doc_col=['user_tags','model'],
                    non_doc_col_append=[])

    
    train_df = train[['instance_id']].copy()
    test_df = test[['instance_id']].copy()

    train_save,test_save,cv_,ho_,ta_ = nn_K_fold(
                                          get_nn_model,
                                          train_fold_dict,
                                          val_fold_dict,
                                          train_fold_y,
                                          val_fold_y,
                                          test_input_dict,
                                          val_index_list,
                                          train_df,
                                          test_df,
                                          pred_col_name = 'predicted_score',
                                          holdout_input_dict=holdout_input_dict,
                                          holdout_y=holdout_y,
                                          holdout_index_list=holdout_index,
                                          nondoc_cols=non_doc_col,
                                          doc_cols=doc_col,
                                          tolerance=0,
                                          preds_batch=5000)
    save_oof(train_save,test_save,cv_,ta_,ho_,file_name='tttt',path=os.path.join(os.path.dirname(__file__),'../../data/nn_ebd/'))