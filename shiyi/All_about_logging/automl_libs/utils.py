import logging
module_logger = logging.getLogger(__name__)

class Utils:
    def __init__(self):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

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
            filename = 'train__' + name + '.pkl'
            df.reset_index(drop=True).to_pickle(url + filename)
            self.logger.info('{} saved at {}'.format(filename, url))
        elif flg == 'test':
            df.reset_index(drop=True).to_pickle(url + 'test__' + name + '.pkl')
        else:
            df[:train_len].reset_index(drop=True).to_pickle(url + 'train__' + name + '.pkl')
            df[train_len:].reset_index(drop=True).to_pickle(url + 'test__' + name + '.pkl')



class FirstClass:
    def __init__(self):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.current_number = 0

    def increment_number(self):
        self.current_number += 1
        self.logger.warning('Incrementing number!')
        self.logger.info('Still incrementing number!!')

    def clear_number(self):
        self.current_number = 0
        self.logger.warning('Clearing number!')
        self.logger.info('Still clearing number!!')


class ThirdClass:
    def __init__(self):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.logger.info('creating an instance of Auxiliary')

    def do_something(self):
        self.logger.info('doing something')
        a = 1 + 1
        self.logger.info('done doing something')


def some_function():
    module_logger.info('received a call to "some_function"')
