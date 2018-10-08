from enum import Enum
import sys,os

class FILE(Enum):
    train_ori = os.path.join(os.path.dirname(__file__),'../../data/original/train.pkl')
    test_ori = os.path.join(os.path.dirname(__file__),'../../data/original/test.pkl')
    train_fe_bf_agg = os.path.join(os.path.dirname(__file__),'../../data/features/train_fe_bf_agg.pkl')
    test_fe_bf_agg = os.path.join(os.path.dirname(__file__),'../../data/features/test_fe_bf_agg.pkl')
    
    train_index = os.path.join(os.path.dirname(__file__),'../../data/original/train_index.pkl')
    train_CV_label = os.path.join(os.path.dirname(__file__),'../../data/original/train_CV_label.pkl')
    train_7_fold_index = os.path.join(os.path.dirname(__file__),'../../data/original/split_idx_by_day.pkl')
    holdout_index = os.path.join(os.path.dirname(__file__),'../../data/original/holdout_index.pkl')
    
    
    
    shiyi_fillna_ori = os.path.join(os.path.dirname(__file__),'../../data/original/all_data_clean1.pkl')
    
    train_base_line = os.path.join(os.path.dirname(__file__),'../../data/features/train_base_line.pkl')
    test_base_line = os.path.join(os.path.dirname(__file__),'../../data/features/test_base_line.pkl')
    
    X_fe_agg_count_formater =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_agg_{}_count.pkl')
    X_fe_agg_time_next_formater =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_agg_{}_time_next.pkl')
    X_fe_agg_time_count_formater =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_agg_{}_time_count.pkl')
    
    X_fe_agg_count_formaterV3 =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_aggV3_{}_count.pkl')
    X_fe_agg_time_next_formaterV3 =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_aggV3_{}_time_next.pkl')
    X_fe_agg_time_count_formaterV3 =  os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_aggV3_{}_time_count.pkl')
    
    X_fe_train_libfm = os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_train_lbfm.pkl')
    X_fe_test_libfm = os.path.join(os.path.dirname(__file__),'../../data/features/X_fe_test_lbfm.pkl')
    X_fe_hashing_format = os.path.join(os.path.dirname(__file__),'../../data/features/feature_hashing/X_fe_hashing_{}.pkl')
    X_fe_emd_format = os.path.join(os.path.dirname(__file__),'../../data/features/feature_ebd/X_fe_ebd_{}.pkl')
    
    train_agg_v1 = os.path.join(os.path.dirname(__file__),'../../data/features/train_v1.pkl')
    test_agg_v1 = os.path.join(os.path.dirname(__file__),'../../data/features/test_v1.pkl')
    train_agg_v2 = os.path.join(os.path.dirname(__file__),'../../data/features/train_v2.pkl')
    test_agg_v2 = os.path.join(os.path.dirname(__file__),'../../data/features/test_v2.pkl')
    train_agg_v3 = os.path.join(os.path.dirname(__file__),'../../data/features/train_v3.pkl')
    test_agg_v3 = os.path.join(os.path.dirname(__file__),'../../data/features/test_v3.pkl')
    train_agg_v3_mean = os.path.join(os.path.dirname(__file__),'../../data/features/train_v3_mean.pkl')
    test_agg_v3_mean = os.path.join(os.path.dirname(__file__),'../../data/features/test_v3_mean.pkl')
    
    train_agg_v1_select = os.path.join(os.path.dirname(__file__),'../../data/features/train_v1_select.pkl')
    test_agg_v1_select = os.path.join(os.path.dirname(__file__),'../../data/features/test_v1_select.pkl')
    train_agg_v2_select = os.path.join(os.path.dirname(__file__),'../../data/features/train_v2_select.pkl')
    test_agg_v2_select = os.path.join(os.path.dirname(__file__),'../../data/features/test_v2_select.pkl')
    train_agg_v3_select = os.path.join(os.path.dirname(__file__),'../../data/features/train_v3_select.pkl')
    test_agg_v3_select = os.path.join(os.path.dirname(__file__),'../../data/features/test_v3_select.pkl')
    
    
    train_appends = os.path.join(os.path.dirname(__file__),'../../data/features/train_appends.pkl')
    test_appends = os.path.join(os.path.dirname(__file__),'../../data/features/test_appends.pkl')
    
    countVectorize_format = os.path.join(os.path.dirname(__file__),'../../data/nn/matrix/countVec_{}.pkl')
    tfidfVectorize_format = os.path.join(os.path.dirname(__file__),'../../data/nn/matrix/tfidfVec_{}.pkl')
    Vectorize_label_format = os.path.join(os.path.dirname(__file__),'../../data/nn/label/label_{}.pkl')
    Vectorize_index_format = os.path.join(os.path.dirname(__file__),'../../data/nn/index/index_{}.pkl')
    
    
    train_lgb_base = os.path.join(os.path.dirname(__file__),'../../data/lgb_feature/train_lgb_baseline.pkl')
    test_lgb_base = os.path.join(os.path.dirname(__file__),'../../data/lgb_feature/test_lgb_baseline.pkl')
    