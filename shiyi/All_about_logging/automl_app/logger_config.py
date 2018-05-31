import logging
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
    logger_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-45s| %(funcName)-20s | #%(lineno)-3d | %(message)s')
    filelog_handler.setFormatter(logger_formatter)
    console_handler.setFormatter(logger_formatter)
    # Add the Handler to the Logger
    logger.addHandler(filelog_handler)
    logger.addHandler(console_handler)
    # logger.info('Completed configuring logger()!')