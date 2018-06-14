import numpy as np
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
        module_logger.debug('{} saved at {}'.format(filename1, url))
        module_logger.debug('{} saved at {}'.format(filename2, url))
    elif flg=='train' or flg=='test':
        filename = flg + '__' + name + '.pkl'
        df.reset_index(drop=True).to_pickle(url + filename)
        module_logger.debug('{} saved at {}'.format(filename, url))
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

class FirstClass:
    def __init__(self):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.current_number = 0

    def increment_number(self):
        self.current_number += 1
        module_logger.warning('Incrementing number!')
        module_logger.info('Still incrementing number!!')

    def clear_number(self):
        self.current_number = 0
        module_logger.warning('Clearing number!')
        module_logger.info('Still clearing number!!')


def do_something(self):
    module_logger.info('doing something')
    a = 1 + 1
    module_logger.info('done doing something')


def some_function():
    module_logger.info('received a call to "some_function"')
