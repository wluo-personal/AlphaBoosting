import logging
module_logger = logging.getLogger(__name__)

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
