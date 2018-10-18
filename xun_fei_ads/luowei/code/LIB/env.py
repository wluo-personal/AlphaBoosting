from enum import Enum
import sys,os

class FILE(Enum):
    train_ori = os.path.join(os.path.dirname(__file__),'../../data/original/train.pkl')
    test_ori = os.path.join(os.path.dirname(__file__),'../../data/original/test.pkl')
    
    train_final = os.path.join(os.path.dirname(__file__),'../../data/original/train_ori_comb_hour.pkl')
    test_final = os.path.join(os.path.dirname(__file__),'../../data/original/test_ori_r2_hour.pkl')
    
    one_hot_base_train = os.path.join(os.path.dirname(__file__),'../../data/features/one_hot/base_train.pkl')
    one_hot_base_test = os.path.join(os.path.dirname(__file__),'../../data/features/one_hot/base_test.pkl')
    
    one_hot_train_formatter = os.path.join(os.path.dirname(__file__),'../../data/features/one_hot/train_{}.pkl')
    one_hot_test_formatter = os.path.join(os.path.dirname(__file__),'../../data/features/one_hot/test_{}.pkl')
    
    X_fe_agg_count_formater = os.path.join(os.path.dirname(__file__),'../../data/features/agg/X_count_agg{}.pkl')
    X_fe_agg_time_next_formater = os.path.join(os.path.dirname(__file__),'../../data/features/agg/X_timeNext_agg{}.pkl')
    X_fe_agg_time_count_formater = os.path.join(os.path.dirname(__file__),'../../data/features/agg/X_timeCount_agg{}.pkl')
    
    X_fe_train_libfm = os.path.join(os.path.dirname(__file__),'../../data/features/agg/X_train_libfm.pkl')
    X_fe_test_libfm = os.path.join(os.path.dirname(__file__),'../../data/features/agg/X_test_libfm.pkl')

    