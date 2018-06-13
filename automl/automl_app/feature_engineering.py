import logging
from automl_libs import FirstClass, ThirdClass, SecondClass
from automl_libs.utils import some_function

logger = logging.getLogger('automl_app.'+__name__)

def do_fe():
    number = FirstClass()
    number.increment_number()
    number.increment_number()
    logger.debug("Current number: %s" % str(number.current_number))
    number.clear_number()
    logger.debug("Current number: %s" % str(number.current_number))
