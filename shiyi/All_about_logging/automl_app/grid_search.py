import logging
from automl_libs import FirstClass, ThirdClass, SecondClass
from automl_libs.utils import some_function

logger = logging.getLogger('automl_app.'+__name__)

def do_gs():
    system = SecondClass()
    system.enable_system()
    system.disable_system()
    logger.debug("Current system configuration: %s" % str(system.enabled))

    third = ThirdClass()
    some_function()
