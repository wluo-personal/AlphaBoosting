import pandas as pd
import numpy as np
from automl_libs import utils
import logging, gc
module_logger = logging.getLogger(__name__)

import pdb
    
# params['train_len'], params['split_col'], dummy_col
def woe(df, cols, dummy_col, generated_feature_name, params=None):
    return _wrapper(df, cols, dummy_col, generated_feature_name,\
                              {'train_len': params['train_len'],\
                               'function': _woe,\
                               'split_col': params['split_col'],\
                               'col': dummy_col})

def chi_square(df, cols, dummy_col, generated_feature_name, params=None):
    return _wrapper(df, cols, dummy_col, generated_feature_name,\
                              {'train_len': params['train_len'],\
                               'function': _chi_square,\
                               'split_col':params['split_col'],\
                               'col': dummy_col})

def mean(df, cols, dummy_col, generated_feature_name, params=None):
    return _wrapper(df, cols, dummy_col, generated_feature_name,\
                              {'train_len': params['train_len'],\
                               'function': _mean,\
                               'split_col':params['split_col'],\
                               'col': dummy_col})

def _wrapper(df, cols, dummy_col, generated_feature_name, params=None):
    train = df[ : params['train_len']]
    test = df[params['train_len']:]
    return pd.concat([_train_wrapper(df[:params['train_len']],\
                                      cols, dummy_col,\
                                      generated_feature_name, params['function'],\
                                      params['split_col']),\
                      _testset_wrapper(df[:params['train_len']],\
                                     df[params['train_len']:],\
                                     cols, dummy_col,\
                                     generated_feature_name, params['function'])],\
                     ignore_index=True)

def _train_wrapper(df, group_cols, label, generated_feature_name, func, split_col):
    r_list = []
    for i in range(df[split_col].min(), df[split_col].max() + 1):
        dictionary = func(df=df[df[split_col]!=i], group_cols=group_cols, label=label, generated_feature_name=generated_feature_name)
        r_list.append(df[df[split_col]==i].merge(dictionary, on=group_cols, how='left')[[generated_feature_name]])
    r = pd.concat(r_list).fillna(-1).reset_index(drop=True)
    del r_list, dictionary
    gc.collect()
    return r

def _testset_wrapper(train, test, group_cols, label, generated_feature_name, func):
    dictionary = func(df=train, group_cols=group_cols, label=label, generated_feature_name=generated_feature_name)
    _df = test.merge(dictionary, on=group_cols, how='left')
    r = _df[[generated_feature_name]].copy().fillna(-1)
    del _df, dictionary
    gc.collect()
    return r

def _woe(df=None, group_cols=None, label=None, generated_feature_name=None, params=None):
    d_cols = list(group_cols)
    d_cols.append(label)
    group = df[d_cols].groupby(by=group_cols)[[label]].agg(['count','sum'])[label].reset_index()
    positive = df[label].sum()
    negative = df.shape[0] - positive
    group[generated_feature_name] = np.log((group['sum']+0.5) / positive / ((group['count']-group['sum']+0.5) / negative))
    dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
    dtype[generated_feature_name] = utils.set_type(group[generated_feature_name], 'float')
    group.astype(dtype)
    return_cols = list(group_cols)
    return_cols.append(generated_feature_name)
    r = group[return_cols]
    del d_cols, group, positive, negative, dtype, return_cols
    gc.collect()
    return r

def _chi_square(df=None, group_cols=None, label=None, generated_feature_name=None, params=None):
    total_count = df.shape[0]
    total_sum = df[label].sum()
    group = df.groupby(by=group_cols)[[label]].agg(['count','sum'])[label].reset_index().rename(index=str, columns={'sum': 'n11'})
    group['n12'] = group['count'] - group['n11']
    group['n21'] = total_sum - group['n11']
    group['n22'] = total_count - group['n11'] - group['n12'] - group['n21']
    group['e11'] = (group['n11'] + group['n12']) * (group['n11'] + group['n21']) / total_count
    group['e12'] = (group['n11'] + group['n12']) * (group['n12'] + group['n22']) / total_count
    group['e21'] = (group['n21'] + group['n22']) * (group['n11'] + group['n21']) / total_count
    group['e22'] = (group['n21'] + group['n22']) * (group['n12'] + group['n22']) / total_count
    group[generated_feature_name] = (group['n11'] - group['e11']) ** 2 / group['e11'] + \
                          (group['n12'] - group['e12']) ** 2 / group['e12'] + \
                          (group['n21'] - group['e21']) ** 2 / group['e21'] + \
                          (group['n22'] - group['e22']) ** 2 / group['e22']
    dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
    dtype[generated_feature_name] = utils.set_type(group[generated_feature_name], 'float')
    group.astype(dtype)
    return_cols = list(group_cols)
    return_cols.append(generated_feature_name)
    r = group[return_cols]
    del group, total_count, total_sum, dtype, return_cols
    gc.collect()
    return r

def _mean(df=None, group_cols=None, label=None, generated_feature_name=None, params=None):
    r = df.groupby(by=group_cols)[[label]].mean().reset_index().rename(index=str, columns={label:generated_feature_name})
    r.astype(utils.set_type(r[generated_feature_name], 'float'))
    gc.collect()
    return r
