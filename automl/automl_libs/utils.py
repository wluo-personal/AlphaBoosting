import numpy as np
import pandas as pd
import random
import string
import os
import logging
module_logger = logging.getLogger(__name__)


def set_type(series, dtype):
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
    """
    flg: str. options: both/train/test
    """
    if flg=='both':
        filename1 = 'train__' + name + '.pkl'
        filename2 = 'test__' + name + '.pkl'
        df[:train_len].reset_index(drop=True).to_pickle(url + filename1)
        df[train_len:].reset_index(drop=True).to_pickle(url + filename2)
        #module_logger.debug('{} saved at {}'.format(filename1, url))
        #module_logger.debug('{} saved at {}'.format(filename2, url))
    elif flg=='train' or flg=='test':
        filename = flg + '__' + name + '.pkl'
        df.reset_index(drop=True).to_pickle(url + filename)
        #module_logger.debug('{} saved at {}'.format(filename, url))
    else:
        raise ValueError('flg options: both/train/test')


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


def get_random_string(length=5):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def save_params_and_result(run_id, model_name, data_name, gs_sn_flag, param_res_dict, keys_to_be_popped, save_dir):
    """
    :param run_id: (str) unique id for the current grid search or stacknet run. (e.g. 3fzG)
    :param model_name: (str) lgb, catb, xgb, nn, etc..
    :param data_name: (str) e.g. w2v_newcat
    :param gs_sn_flag: (str) either "gridsearch" or "stacknet"
    :param param_res_dict: (dict) combined parameters and grid search / stacknet results (metrics)
    :param keys_to_be_popped: (list of str) keys to be popped out the param_res_dict
    :param save_dir: (str) directory to save the param_res_dict as Dataframe
    :return: None
    """
    param_res_dict['data_name'] = data_name
    # get the param_res_dict ready for storage
    for key in keys_to_be_popped:
        param_res_dict.pop(key, None)
    # so that [1,2,3] can be converted to "[1,2,3]" and be treated as a whole in csv
    for k, v in param_res_dict.items():
        if isinstance(v, list):
            param_res_dict[k] = '"' + str(v) + '"'
            # module_logger.debug(param_res_dict[k])

    res = pd.DataFrame(param_res_dict, index=[run_id])
    filename_for_gs_results = save_dir + '{}_{}_{}.csv'.format(model_name, data_name, gs_sn_flag)
    if not os.path.exists(filename_for_gs_results):
        res.to_csv(filename_for_gs_results)
        module_logger.debug(filename_for_gs_results + ' created')
    else:
        old_res = pd.read_csv(filename_for_gs_results, index_col='Unnamed: 0')
        res = pd.concat([old_res, res])
        res.to_csv(filename_for_gs_results)
        module_logger.info(filename_for_gs_results + ' updated')