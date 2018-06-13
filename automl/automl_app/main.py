import feature_engineering as fe
import grid_search as gs
import logger_config, logging
logger_config.config('automl.log')
logger = logging.getLogger('AutoML.automl_app.app')

logger.info('start feature engineering...')
fe.do_fe()
logger.info('feature engineering done')

logger.info('start grid search...')
gs.do_gs()
logger.info('grid search done')
