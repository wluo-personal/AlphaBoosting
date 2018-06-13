
# coding: utf-8

# In[1]:


import pdb


# In[2]:


import pandas as pd
import os
import re
import json
import gc
import shutil


# # FE

# In[3]:


import pandas as pd
import numpy as np
from collections import defaultdict
import gc

class Feature:
    
    class Utils:
        def _set_type(series, dtype):
            """Returns datatype with appropriate data size.
            Appropriate data size is decided by checking minimum and maximum values in the series.

            Parameters
            ----------
            series : series
                The series of data that datatype needs to be modified.
            dtype : str
                The string of datatype name.        
            """              
            _max, _min = max(series), min(series)
            if dtype == 'uint':
                if _max <= 255: return np.uint8
                elif _max <= 65535: return np.uint16
                elif _max <= 4294967295: return np.uint32
                else: return np.uint64
            elif dtype == 'int':
                if _min >= -128 and _max <= 127: return np.int8
                elif _min >=-32768 and _max <= 32767: return np.int16
                elif _min >= -2147483648 and _max <= 2147483647: return np.int32
                else: return np.int64
            elif dtype == 'float':
                if max(abs(_min), _max) <= 3.4028235e+38: return np.float32
                else: return np.float64

        def save(df=None, flg='both', train_len=0, url='./', name='default'):
            if flg == 'train':
                df.reset_index(drop=True).to_pickle(url + 'train__' + name + '.pkl')
            elif flg == 'test':
                df.reset_index(drop=True).to_pickle(url + 'test__' + name + '.pkl')
            else:
                df[:train_len].reset_index(drop=True).to_pickle(url + 'train__' + name + '.pkl')
                df[train_len:].reset_index(drop=True).to_pickle(url + 'test__' + name + '.pkl')
    
    
    
    # params['col']
    def count(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        count number of each values in specified column in input dataframe.
        If specified columns are multiple, 
        then count number of different combination of values would be returned.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns that to be counted.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, only params['col'] is used.
            params['col'] is a string of column name, and this column is usually used to aid calculation.
            
        Example
        -------
        df:
            a	b	label
        0	0	0	1
        1	0	1	1
        2	0	0	1
        3	0	1	0
        4	1	1	1
        5	1	1	0
        6	1	1	1
        7	1	0	0
        
        call: count(df, cols=['a','b'], col_name='count_a_b', params={'col':'label'})
        
        returns:
            count_a_b
        0	2
        1	2
        2	2
        3	2
        4	3
        5	3
        6	3
        7	1

            
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        """    
        r_col = params['col']
        dtype = {x: df[x].dtype for x in cols if x in df.columns.values}
        d_cols = list(cols)
        d_cols.append(r_col)
        result = df[d_cols].groupby(by=cols)[[r_col]].count().rename(index=str, columns={r_col: col_name}).reset_index()
        dtype[col_name] = Feature.Utils._set_type(result[col_name], 'uint')
        _df = df.merge(result.astype(dtype), on=cols, how='left')
        r = _df[[col_name]].copy()
        del _df, result, d_cols, dtype
        gc.collect()
        return r
    
    def unique_count(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        count number of unique values of a spesific column 
        in 'a specific value of another column' or 'a specific combinations of other columns'.
        
        cols[:-1] are grouped into unique combinations, and given a unique combination,
        the number of unique values in cols[-1] would be counted.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
            The last column in the array is the one that the unique values are counted.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params is not used.
            
        Example
        -------
        df:
            a	b	label
        0	0	0	1
        1	0	1	1
        2	0	2	1
        3	0	3	0
        4	1	1	1
        5	1	1	0
        6	1	1	1
        7	1	1	0 
        
        call: unique_count(df, cols=['a','b'], col_name='unique_count_a_b', params={'col':'label'})
        
        returned:
            unique_count_a_b
        0	4
        1	4
        2	4
        3	4
        4	1
        5	1
        6	1
        7	1
            
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        """ 
        r_col = cols[-1]
        dtype = {x: df[x].dtype for x in cols[:-1] if x in df.columns.values}
        result = df[cols].groupby(by=cols[:-1])[[r_col]].nunique().rename(index=str, columns={r_col: col_name}).reset_index()
        dtype[col_name] = Feature.Utils._set_type(result[col_name], 'uint')
        _df = df.merge(result.astype(dtype), on=cols[:-1], how='left')
        r = _df[[col_name]].copy()
        del _df, result, dtype
        gc.collect()
        return r
    
    def cumulative_count(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        cumulative count number of 
        'unique values of a spesified column' or 'unique combinations of spesified columns'.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params is not used.
            
        Example
        -------
        df:
            a	b	label
        0	0	0	1
        1	0	1	1
        2	1	2	1
        3	0	3	0
        4	1	1	1
        5	2	1	0
        6	0	1	1
        7	1	1	0
        
        call: cumulative_count(df, cols=['a'], col_name='cum_count_a_b', params={'col':'label'})
        
        returned:
            cum_count_a_b
        0	0
        1	1
        2	0
        3	2
        4	1
        5	0
        6	3
        7	2

        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        reversed_cumulative_count : reversed version
        """
        result = df[cols].groupby(by=cols).cumcount().rename(col_name)
        r = result.astype(Feature.Utils._set_type(result, 'uint'))
        r = r.to_frame()
        del result
        gc.collect()
        return r
    
    def reverse_cumulative_count(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        cumulative count number of 
        'unique values of a spesified column' or 'unique combinations of spesified columns'
        in reversed order.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params is not used.
            
        Example
        -------
        df:
            a	b	label
        0	0	0	1
        1	0	1	1
        2	1	2	1
        3	0	3	0
        4	1	1	1
        5	2	1	0
        6	0	1	1
        7	1	1	0
        
        call: reverse_cumulative_count(df, cols=['a'], col_name='rev_cum_count_a_b', params={'col':'label'})
        
        returned:
            rev_cum_count_a_b
        0	3
        1	2
        2	2
        3	1
        4	1
        5	0
        6	0
        7	0
        
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        cumulative_count : unreversed version
        """        
        result = df.sort_index(ascending=False)[cols].groupby(cols).cumcount().rename(col_name)
        r = result.astype(Feature.Utils._set_type(result, 'uint')).to_frame()
        r.sort_index(inplace=True)
        del result
        gc.collect()
        return r
    
    def variance(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        variance of a specified column given the unique combinations of other columns.
        
        cols[:-1] are grouped into unique combinations, and given a unique combination,
        variance of values in cols[-1] would be calculated.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params is not used.
        
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        """ 
        group_cols = cols[:-1]
        calc_col = cols[-1]
        group = df[cols].groupby(by=group_cols)[[calc_col]].var().reset_index().rename(index=str, columns={calc_col: col_name}).fillna(0)
        dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
        dtype[col_name] = Feature.Utils._set_type(group[col_name], 'float')
        _df = df.merge(group.astype(dtype), on=group_cols, how='left')
        r = _df[[col_name]].copy()
        del dtype, _df, group
        gc.collect()
        return r
    
    # params['col'] = : additional col to help count
    # params['coefficient']: 
    def count_std_over_mean(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        (coefficient * standard deviation)/mean of count number of combinations.
        
        cols are grouped into unique combinations,
        and count number of each combination will be added in 'count' column.
        Then, cols[:-1] would be grouped into unique combinations, 
        and standard deviation and mean of 'count' in each unique combinations would be calculated 
        and recorded as 'std' and 'mean'.      
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params['col'] and params['coefficient'] are used.
            params['col'] is a string of a column name, and this column is usually used to aid the calculation.
            params['coefficient'] is a string of a number.
     
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        """ 
        group_cols = cols[:-1]
        calc_col = cols[-1]
        d_cols = list(cols)
        d_cols.append(params['col'])
        group = df[d_cols].groupby(by=cols)[[params['col']]].count().reset_index().rename(index=str, columns={params['col']: 'count'})
        result = group.groupby(by=group_cols)[['count']].agg(['mean','std'])['count'].reset_index()
        result[col_name] = ((int(params['coefficient']) * result['std']) / result['mean']).fillna(-1)
        dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
        dtype[col_name] = Feature.Utils._set_type(result[col_name], 'float')
        _df = df.merge(result.astype(dtype), on=group_cols, how='left')
        r = _df[[col_name]].copy()
        del d_cols, group, result, _df
        gc.collect()
        return r
    
    
    
    
    # params['n']: n, params['fillna']: fillna, cols[-1]: time
    def time_to_n_next(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        integers that indicates the time needed to reach the nth next occurrence of the specific value.
        n=1 indicates the next occurence.
        n=2 indicates the next of next occurence.
        
        cols[:-1] are grouped into unique combinations and the time column is shifted based on unique combinations,
        and the shifted time is subtracted by the original time 
        to determine the time needed to reach the nth next occurrence.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
            cols[-1] must be a column that indicates time.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params['n'] and params['fillna'] are used.
            params['n'] is a string of number that indicates the nth next occurrence.
            params['fillna'] is a string of number that specifies the number to replace NaN.
            
        Example
        -------
        n=1
        fillna=222
        
        df:
            a	label	t
        0	0	1	0
        1	1	1	1
        2	1	1	2
        3	1	0	3
        4	2	1	4
        5	0	0	5
        6	1	1	6
        7	0	0	7
        
        call: time_to_n_next(df, cols=['a','t'], col_name='time_to_n_next', params={'n':'1', 'fillna': '222'})
        
        returned:
            t
        0	5
        1	1
        2	1
        3	3
        4	222
        5	2
        6	222
        7	222
        
        
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        
        Note
        -----
        pandas.DataFrame.groupby().shift() : shifts within the unique values/combinations of cols[:-1]
        """         
        group_cols = cols[:-1]
        calc_col = cols[-1]
        n = int(params['n'])
        m = int(params['fillna'])
        result = (df[cols].groupby(by=group_cols)[calc_col].shift(-n) - df[calc_col]).fillna(m)
        result = result.astype(Feature.Utils._set_type(result, 'uint')).to_frame()
        del n, m
        gc.collect()
        return result
    
    # params['n']: n, cols[-1]: time
    def count_in_previous_n_time_unit(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        integers that indicates the number of previous occurrences of specific values
        within the time n before the current time.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
            cols[-1] must be a column that indicates time.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params['n'] is used.
            params['n'] is a string of number that indicates the number of time units that need to go back.
        
        Example
        -------
        n=3
        
        df:
            a	label	t
        0	0	1	0
        1	1	1	1
        2	1	1	2
        3	1	0	3
        4	1	1	4
        5	0	0	5
        6	1	1	6
        7	0	0	7        
        
        call: count_in_previous_n_time_unit(df, cols=['a','t'], col_name='count_prev_n', params={'col':'label', 'n':'3', 'fillna': '222'})
        
        returned:
            count_prev_n
        0	0
        1	0
        2	1
        3	2
        4	3
        5	0
        6	2
        7	1        
        
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        count_in_next_n_time_unit : counts the occurrences of specific values with in the time n after the current time.
        """        
        group_cols = cols[:-1]
        calc_col = cols[-1]
        n = int(params['n'])
        encodings = df[group_cols[0]].copy()
        if len(group_cols) > 1:
            for c in group_cols[1 : ]:
                encodings = encodings * (10 ** (int(np.log(df[c].max() + 1) / np.log(10)) + 1)) + df[c]
        encodings = encodings.values
        times = df[calc_col].values
        dict_count = defaultdict(int)
        result = []
        bound = 0
        for cur in range(len(encodings)):
            while abs(times[cur] - times[bound]) > n:
                dict_count[encodings[bound]] -= 1
                bound += 1
            result.append(dict_count[encodings[cur]])
            dict_count[encodings[cur]] += 1
        r = pd.DataFrame(result, columns=[col_name], dtype=Feature.Utils._set_type(result, 'uint'))
        del encodings, times, dict_count, result, bound, n
        gc.collect()
        return r
    
    # cols[-1]: time
    def count_in_next_n_time_unit(df=None, cols=None, col_name=None, params=None):
        """Returns dataframe of one feature that consist of 
        integers that indicates the number of previous occurrences of specific values
        within the time n after the current time.
        
        Parameters
        ----------
        df : dataframe, shape (n_samples, n_features)
            The data.
        cols : array-like
            Array of string names of columns.
            cols[-1] must be a column that indicates time.
        col_name : str
            This will be the name of column in the returned dataframe.
        params : dictionary
            Params is a dictionary that has various parametors.
            In this method, params['n'] is used.
            params['n'] is a string of number that indicates the number of time units that need to go back.
            
        Example
        -------
        n=3
        
        df:
            a	label	t
        0	0	1	0
        1	1	1	1
        2	1	1	2
        3	1	0	3
        4	1	1	4
        5	0	0	5
        6	1	1	6
        7	0	0	7    
        
        call: count_in_next_n_time_unit(df, cols=['a','t'], col_name='count_next_n', params={'n':'3'})
        
        returned:
            count_next_n
        0	0
        1	3
        2	2
        3	2
        4	1
        5	1
        6	0
        7	0        
        
        See Also
        --------
        Utils._set_type : This is used to set suited data type to the column of dataframe that will be returned.
        count_in_previous_n_time_unit : 
            counts the occurrences of specific values with in the time n before the current time.
        """          
        r = Feature.count_in_previous_n_time_unit(df.sort_index(ascending=False), cols, col_name, params)
        r = r.reindex(index=r.index[::-1]).reset_index(drop=True)
        gc.collect()
        return r
    
    
    
    class Encoding:
        # params['trainLen'], params['splitCol'], params['col']
        def woe(df=None, cols=None, col_name=None, params=None):
            return Feature.Encoding._wrapper(df, cols, col_name,                                      {'train_len': params['trainLen'],                                       'function': Feature.Encoding._woe,                                       'split_col': params['splitCol'],                                       'col': params['col']})
        
        def chi_square(df=None, cols=None, col_name=None, params=None):
            return Feature.Encoding._wrapper(df, cols, col_name,                                      {'train_len': params['trainLen'],                                       'function': Feature.Encoding._chi_square,                                       'split_col':params['splitCol'],                                       'col': params['col']})
        
        def mean(df=None, cols=None, col_name=None, params=None):
            return Feature.Encoding._wrapper(df, cols, col_name,                                      {'train_len': params['trainLen'],                                       'function': Feature.Encoding._mean,                                       'split_col':params['splitCol'],                                       'col': params['col']})
        
        def _wrapper(df=None, cols=None, col_name=None, params=None):
            train = df[ : params['train_len']]
            test = df[params['train_len']:]
            return pd.concat([Feature.Encoding._train_wrapper(df[:params['train_len']],                                                              cols, params['col'],                                                              col_name, params['function'],                                                              params['split_col']),                              Feature.Encoding._testset_wrapper(df[:params['train_len']],                                                             df[params['train_len']:],                                                             cols, params['col'],                                                             col_name, params['function'])],                             ignore_index=True)
        
        def _train_wrapper(df, group_cols, label, col_name, func, split_col):
            r_list = []
            for i in range(df[split_col].min(), df[split_col].max() + 1):
                dictionary = func(df=df[df[split_col]!=i], group_cols=group_cols, label=label, col_name=col_name)
                r_list.append(df[df[split_col]==i].merge(dictionary, on=group_cols, how='left')[[col_name]])
            r = pd.concat(r_list).fillna(-1).reset_index(drop=True)
            del r_list, dictionary
            gc.collect()
            return r
        
        def _testset_wrapper(train, test, group_cols, label, col_name, func):
            dictionary = func(df=train, group_cols=group_cols, label=label, col_name=col_name)
            _df = test.merge(dictionary, on=group_cols, how='left')
            r = _df[[col_name]].copy().fillna(-1)
            del _df, dictionary
            gc.collect()
            return r
        
        def _woe(df=None, group_cols=None, label=None, col_name=None, params=None):
            d_cols = list(group_cols)
            d_cols.append(label)
            group = df[d_cols].groupby(by=group_cols)[[label]].agg(['count','sum'])[label].reset_index()
            positive = df[label].sum()
            negative = df.shape[0] - positive
            group[col_name] = np.log((group['sum']+0.5) / positive / ((group['count']-group['sum']+0.5) / negative))
            dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
            dtype[col_name] = Feature.Utils._set_type(group[col_name], 'float')
            group.astype(dtype)
            return_cols = list(group_cols)
            return_cols.append(col_name)
            r = group[return_cols]
            del d_cols, group, positive, negative, dtype, return_cols
            gc.collect()
            return r
        
        def _chi_square(df=None, group_cols=None, label=None, col_name=None, params=None):
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
            group[col_name] = (group['n11'] - group['e11']) ** 2 / group['e11'] +                                   (group['n12'] - group['e12']) ** 2 / group['e12'] +                                   (group['n21'] - group['e21']) ** 2 / group['e21'] +                                   (group['n22'] - group['e22']) ** 2 / group['e22']
            dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
            dtype[col_name] = Feature.Utils._set_type(group[col_name], 'float')
            group.astype(dtype)
            return_cols = list(group_cols)
            return_cols.append(col_name)
            r = group[return_cols]
            del group, total_count, total_sum, dtype, return_cols
            gc.collect()
            return r
        
        def _mean(df=None, group_cols=None, label=None, col_name=None, params=None):
            r = df.groupby(by=group_cols)[[label]].mean().reset_index().rename(index=str, columns={label:col_name})
            r.astype(Feature.Utils._set_type(r[col_name], 'float'))
            gc.collect()
            return r
            
        
        
    class Kernels:
        def square(df=None, cols=None, col_name=None, params=None):
            r = df[[cols]].apply(lambda x: x ** 2)
            r = r.astype(Feature.Utils._set_type(r, 'float'))
            gc.collect()
            return r
        
func_map = {
        'count':                         Feature.count,
        'unique_count':                  Feature.unique_count,
        'cumulative_count':              Feature.cumulative_count,
        'reverse_cumulative_count':      Feature.reverse_cumulative_count,
        'variance':                      Feature.variance,
        'count_std_over_mean':           Feature.count_std_over_mean,
        'time_to_n_next':                Feature.time_to_n_next,
        'count_in_previous_n_time_unit': Feature.count_in_previous_n_time_unit,
        'count_in_next_n_time_unit':     Feature.count_in_next_n_time_unit,
        'woe':                           Feature.Encoding.woe,
        'chi_square':                    Feature.Encoding.chi_square,
        'mean':                          Feature.Encoding.mean,
        'square':                        Feature.Kernels.square
    }


# # AB

# In[8]:


class AlphaBoosting:
    def __init__(self, root=None, train_csv_url=None, test_csv_url=None, validation_index=None, timestamp=None,
                 label=None, categorical_features=None, numerical_features=None, validation_ratio=0.1, ngram=(1,1),
                 downsampling=1, down_sampling_ratio=None, run_record='run_record.json'):
        downsampling_amount_changed = False
        down_sampling_ratio_changed = False
        val_index_changed = False
        if run_record == None:
            raise Exception('run record file can not be None')
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
                    self._add_column(line, func_map)
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
        # TODO: finish grid search
        self._renew_status(dictionary, 'grid_search', self.LOGDIR + 'todo_list.json')
    
    
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
            Feature.Utils.save(df=_df, train_len=self.train_len, url=self.FEATUREDIR, name=col_name)
    
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


# In[9]:


# a = AlphaBoosting(configuration='./log/config.json')
# a = AlphaBoosting(root='./', train_csv_url='./a.txt', test_csv_url='./b.txt', validation_index=[1,2], label='l',down_sampling_ratio=0.5, downsampling=5, configuration='./log/config.json')

#a = AlphaBoosting(root='./', train_csv_url='./a.txt', test_csv_url='./b.txt', validation_index=[1,2], label='l',down_sampling_ratio=0.5, downsampling=2)

