from automl_libs import utils
import logging
module_logger = logging.getLogger(__name__)


def square(df=None, cols=None, col_name=None, params=None):
    r = df[[cols]].apply(lambda x: x ** 2)
    r = r.astype(utils.set_type(r, 'float'))
    gc.collect()
    return r