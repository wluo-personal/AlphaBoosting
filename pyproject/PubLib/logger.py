import logging
import os,sys
sys.path.append('/home/kai/data/wei/AlphaBoosting/')
from pyproject.ENV.env import ENV

class Logger:
    def __init__(self, logger_name, log_file, pt=False):
        self.logger = logging.getLogger(logger_name)
        self.logLevel = logging.DEBUG
        self.logmapping ={'INFO': logging.INFO, 'DEBUG': logging.DEBUG}
        self._file = log_file
        self.init_log(pt)
        
    def init_log(self, pt):
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s- %(funcName)s -%(lineno)d - %(message)s')
        if pt:
            sh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - P%(process)d - T%(thread)d - %(name)s- %(funcName)s -%(lineno)d- \t%(message)s')
        else:
            sh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - CLASS:%(name)s- METHOD:%(funcName)s -LINE:%(lineno)d - MSG:%(message)s')

        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(sh_formatter)
        fh = logging.FileHandler(self._file)
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        
        
#Usage
# class Test:
#     def __init__(self, log_file):
#         self.log = Logger(self.__class__.__name__, log_file, pt=True).logger

#     def testlog(self, msg):
#         self.log.info(msg)
# t = Test(ENV.log_path)
# t.testlog('tttt')