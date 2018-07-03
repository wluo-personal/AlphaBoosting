import logging, time
from datetime import datetime
from pytz import timezone, utc
from os import path, remove
# If applicable, delete the existing log file to generate a fresh log file during each execution

def config(file_name, file_loglevel=logging.DEBUG, console_loglevel=logging.DEBUG):
    if path.isfile(file_name):
        remove(file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Create the Handler for logging data to a file
    filelog_handler = logging.FileHandler(file_name)
    filelog_handler.setLevel(file_loglevel)
    # Create the Handler for logging data to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_loglevel)
    # Create a Formatter for formatting the log messages
    def customTime(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("US/Eastern")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime
    logger_formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)-20s '
                                         '| #%(lineno)-3d | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")# EST")
    filelog_handler.setFormatter(logger_formatter)
    console_handler.setFormatter(logger_formatter)
    # Add the Handler to the Logger
    logger.addHandler(filelog_handler)
    logger.addHandler(console_handler)
    # logger.info('Completed configuring logger()!')