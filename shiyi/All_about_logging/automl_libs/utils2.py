import logging
class SecondClass(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.enabled = False

    def enable_system(self):
        self.enabled = True
        self.logger.warning('Enabling system!')
        self.logger.info('Still enabling system!!')

    def disable_system(self):
        self.enabled = False
        self.logger.warning('Disabling system!')
        self.logger.info('Still disabling system!!')