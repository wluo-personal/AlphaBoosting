from collections import defaultdict
from automl_libs import utils
from automl_libs import feature_engineering as fe
import pandas as pd
import logging, gc
module_logger = logging.getLogger(__name__)

def count(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, not used here

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

    call: count(df, cols=['a','b'], dummy_col='label', generated_feature_name='count_a_b') 

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    """    
    r_col = dummy_col
    dtype = {x: df[x].dtype for x in cols if x in df.columns.values}
    d_cols = list(cols)
    d_cols.append(r_col)
    result = df[d_cols].groupby(by=cols)[[r_col]].count().rename(index=str, columns={r_col: generated_feature_name}).reset_index()
    dtype[generated_feature_name] = utils.set_type(result[generated_feature_name], 'uint')
    _df = df.merge(result.astype(dtype), on=cols, how='left')
    r = _df[[generated_feature_name]].copy()
    del _df, result, d_cols, dtype
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

def unique_count(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary,  not used here

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

    call: unique_count(df, cols=['a','b'], dummy_col='label', generated_feature_name='unique_count_a_b')

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.

    Notes
    -----
    """ 
    r_col = cols[-1]
    dtype = {x: df[x].dtype for x in cols[:-1] if x in df.columns.values}
    result = df[cols].groupby(by=cols[:-1])[[r_col]].nunique().rename(index=str, columns={r_col: generated_feature_name}).reset_index()
    dtype[generated_feature_name] = utils.set_type(result[generated_feature_name], 'uint')
    _df = df.merge(result.astype(dtype), on=cols[:-1], how='left')
    r = _df[[generated_feature_name]].copy()
    del _df, result, dtype
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

def cumulative_count(df, cols, dummy_col, generated_feature_name, params=None):
    """Returns dataframe of one feature that consist of 
    cumulative count number of 
    'unique values of a spesified column' or 'unique combinations of spesified columns'.

    Parameters
    ----------
    df : dataframe, shape (n_samples, n_features)
        The data.
    cols : array-like
        Array of string names of columns.
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, not used here

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

    call: cumulative_count(df, cols=['a'], dummy_col='label', generated_feature_name='cum_count_a_b')

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    reversed_cumulative_count : reversed version
    """
    result = df[cols].groupby(by=cols).cumcount().rename(generated_feature_name)
    r = result.astype(utils.set_type(result, 'uint'))
    r = r.to_frame()
    del result
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

def reverse_cumulative_count(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, not used here

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

    call: reverse_cumulative_count(df, cols=['a'], generated_feature_name='rev_cum_count_a_b', params=None)

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    cumulative_count : unreversed version
    """        
    result = df.sort_index(ascending=False)[cols].groupby(cols).cumcount().rename(generated_feature_name)
    r = result.astype(utils.set_type(result, 'uint')).to_frame()
    r.sort_index(inplace=True)
    del result
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

def variance(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, not used here

    See Also
    --------
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    """ 
    group_cols = cols[:-1]
    calc_col = cols[-1]
    group = df[cols].groupby(by=group_cols)[[calc_col]].var().reset_index().rename(index=str, columns={calc_col: generated_feature_name}).fillna(0)
    dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
    dtype[generated_feature_name] = utils.set_type(group[generated_feature_name], 'float')
    _df = df.merge(group.astype(dtype), on=group_cols, how='left')
    r = _df[[generated_feature_name]].copy()
    del dtype, _df, group
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

def count_std_over_mean(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, params['coefficient'] are used.
        params['coefficient'] is a string of a number.

    See Also
    --------
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    """ 
    group_cols = cols[:-1]
    calc_col = cols[-1]
    d_cols = list(cols)
    d_cols.append(dummy_col)
    group = df[d_cols].groupby(by=cols)[[dummy_col]].count().reset_index().rename(index=str, columns={dummy_col: 'count'})
    result = group.groupby(by=group_cols)[['count']].agg(['mean','std'])['count'].reset_index()
    result[generated_feature_name] = ((int(params['coefficient']) * result['std']) / result['mean']).fillna(-1)
    dtype = {x: df[x].dtype for x in group_cols if x in df.columns.values}
    dtype[generated_feature_name] = utils.set_type(result[generated_feature_name], 'float')
    _df = df.merge(result.astype(dtype), on=group_cols, how='left')
    r = _df[[generated_feature_name]].copy()
    del d_cols, group, result, _df
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r




# params['n']: n, params['fillna']: fillna, cols[-1]: time
def time_to_n_next(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, params['n'] and params['fillna'] are used.
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

    call: time_to_n_next(df, cols=['a','t'], generated_feature_name='time_to_n_next', params={'n':'1', 'fillna': '222'})

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.

    Note
    -----
    pandas.DataFrame.groupby().shift() : shifts within the unique values/combinations of cols[:-1]
    """         
    group_cols = cols[:-1]
    calc_col = cols[-1]
    n = int(params['n'])
    m = int(params['fillna'])
    result = (df[cols].groupby(by=group_cols)[calc_col].shift(-n) - df[calc_col]).fillna(m)
    result = result.astype(utils.set_type(result, 'uint')).to_frame()
    del n, m
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return result

def count_in_previous_n_time_unit(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, params['n'] is used.
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

    call: count_in_previous_n_time_unit(df, cols=['a','t'], generated_feature_name='count_prev_n', params={'n':'3'})

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
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
    r = pd.DataFrame(result, columns=[generated_feature_name], dtype=utils.set_type(result, 'uint'))
    del encodings, times, dict_count, result, bound, n
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r

# cols[-1]: time
def count_in_next_n_time_unit(df, cols, dummy_col, generated_feature_name, params=None):
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
    generated_feature_name : str
        This will be the name of column in the returned dataframe.
    params : dictionary, params['n'] is used.
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

    call: count_in_next_n_time_unit(df, cols=['a','t'], generated_feature_name='count_next_n', params={'n':'3'})

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
    utils.set_type : This is used to set suited data type to the column of dataframe that will be returned.
    count_in_previous_n_time_unit : 
        counts the occurrences of specific values with in the time n before the current time.
    """          
    r = fe.count_in_previous_n_time_unit(df.sort_index(ascending=False), cols, dummy_col, generated_feature_name, params)
    r = r.reindex(index=r.index[::-1]).reset_index(drop=True)
    gc.collect()
    module_logger.debug('feature generated: {}'.format(generated_feature_name))
    return r