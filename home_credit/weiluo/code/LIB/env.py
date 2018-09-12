from enum import Enum
import sys,os

class ENV(Enum):
    sample_submission_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/sample_submission.csv')
    credit_card_balance_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/credit_card_balance.csv')
    bureau_balance_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/bureau_balance.csv')
    POS_CASH_balance_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/POS_CASH_balance.csv')
    application_test_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/application_test.csv')
    bureau_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/bureau.csv')
    installments_payments_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/installments_payments.csv')
    previous_application_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/previous_application.csv')
    application_train_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/application_train.csv')
    HomeCredit_columns_description_ori = os.path.join(os.path.dirname(__file__),'../../data/original_data/HomeCredit_columns_description.csv')
    
    ####### reproduce
    application_train_reproduce = os.path.join(os.path.dirname(__file__),'../../data/reproduce/feature_engineering/fe_application_train.pkl')
#     application_train_reproduce = os.path.join(os.path.dirname(__file__),'../../data/reproduce/feature_engineering/fe_application_test.pkl')
    application_test_reproduce = os.path.join(os.path.dirname(__file__),'../../data/reproduce/feature_engineering/fe_application_test.pkl')
    
    
    ### fillNA
    application_train_cleaned = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/application_train.pkl')
    application_test_cleaned = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/application_test.pkl')
    
    previous_application_cleaned = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/previous_application.pkl')
    previous_application_cleaned_onehot = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/previous_application_onehot.pkl')
    
    bureau_cleaned = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/bureau.pkl')
    bureau_cleaned_rnnALL = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/bureau_cleaned_rnnALL.pkl')
    
    bureau_balance_clean = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/bureau_balance.pkl')
    bureau_balance_clean_rnn = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/bureau_balance_rnn.pkl')
    
    credit_card_balance_clean = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/credit_card_balance.pkl')
    credit_card_balance_clean_rnn = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/credit_card_balance_rnn.pkl')
    
    installments_payments_clean = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/installments_payments.pkl')
    installments_payment_clean_rnn = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/installments_payments_rnn.pkl')
    
    POS_CASH_balance_clean = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/POS_CASH_balance.pkl')
    POS_CASH_balance_clean_rnn = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/POS_CASH_balance_rnn.pkl')
    
    clean_categorical_col = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/categorical_col.pkl')
    
    
    ############## ALL combine to one file
    previous_app_combine_rnnALL = os.path.join(os.path.dirname(__file__),'../../data/cleaned_data/previous_application_rnn_ALL.pkl')
    
    
    
    
    
    ### RNN
    train_fold_index = os.path.join(os.path.dirname(__file__),'../../data/rnn/train_fold_index.pkl')
    val_fold_index = os.path.join(os.path.dirname(__file__),'../../data/rnn/val_fold_index.pkl')
    
   
    previous_application_rnn = os.path.join(os.path.dirname(__file__),'../../data/rnn/previous_application/fold_{}.hdf5')
    previous_application_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/previous_application/report_fold_{}.pkl')
    previous_application_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/previous_application/preds_fold_{}.pkl')
    previous_application_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/previous_application/test_preds_fold_{}.pkl')

    
    bureau_rnn = os.path.join(os.path.dirname(__file__),'../../data/rnn/bureau/fold_{}.hdf5')
    bureau_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/bureau/report_fold_{}.pkl')
    bureau_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/bureau/preds_fold_{}.pkl')
    bureau_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/bureau/test_preds_fold_{}.pkl')
    
    installment_rnn = os.path.join(os.path.dirname(__file__),'../../data/rnn/installment/fold_{}.hdf5')
    installment_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/installment/report_fold_{}.pkl')
    installment_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/installment/preds_fold_{}.pkl')
    installment_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/installment/test_preds_fold_{}.pkl')
    
    POS_CASH_rnn = os.path.join(os.path.dirname(__file__),'../../data/rnn/poscash/fold_{}.hdf5')
    POS_CASH_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/poscash/report_fold_{}.pkl')
    POS_CASH_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/poscash/preds_fold_{}.pkl')
    POS_CASH_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/poscash/test_preds_fold_{}.pkl')
    
    credit_card_rnn = os.path.join(os.path.dirname(__file__),'../../data/rnn/creditcard/fold_{}.hdf5')
    credit_card_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/creditcard/report_fold_{}.pkl')
    credit_card_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/creditcard/preds_fold_{}.pkl')
    credit_card_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/creditcard/test_preds_fold_{}.pkl')
    
    
    
    ### lightFMM
    lightfm_v1 = os.path.join(os.path.dirname(__file__),'../../data/lightFM/all_v1.pkl')
    
    
    ### lightgbm
    lightgbm_train_764 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_fe_939_corr85_cols764.pkl')
    lightgbm_test_764 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_fe_939_corr85_cols764.pkl')
    
    lightgbm_train_764_nn = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_fe_939_corr85_cols764_nn.pkl')
    lightgbm_test_764_nn = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_fe_939_corr85_cols764_nn.pkl')
    
    
    
    #### NN Plain
    main_plain_nn = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_plain/fold_{}.hdf5')
    main_plain_nn_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_plain/report_fold_{}.pkl')
    main_plain_nn_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_plain/preds_fold_{}.pkl')
    main_plain_nn_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_plain/test_preds_fold_{}.pkl')
    
    
    #### NN ALL
    main_ALL_nn = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_ALL/fold_{}.hdf5')
    main_ALL_nn_report = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_ALL/report_fold_{}.pkl')
    main_ALL_nn_preds = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_ALL/preds_fold_{}.pkl')
    main_ALL_nn_preds_test = os.path.join(os.path.dirname(__file__),'../../data/rnn/main_ALL/test_preds_fold_{}.pkl')
    
    
    ################# feature selection
    drop_column_report = os.path.join(os.path.dirname(__file__),'../../code/gbm_features/dropEachColumnsReport.pkl')
    coff_764 = os.path.join(os.path.dirname(__file__),'../../data/add_features/coff_764.pkl')
    
    
    
    #####lightgbm0827
    lgb_train_0827_raw = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_0827raw.pkl')
    lgb_test_0827_raw = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_0827raw.pkl')
    lgb_train_0827 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_0827.pkl')
    lgb_test_0827 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_0827.pkl')
    lgb_train_0827_na = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_0827na.pkl')
    lgb_test_0827_na = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_0827na.pkl')
    lgb_train_0827_na_extrem = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_0827na_ex.pkl')
    lgb_test_0827_na_extrem = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_0827na_ex.pkl')
    
    lgb_train_0828 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/train_0828.pkl')
    lgb_test_0828 = os.path.join(os.path.dirname(__file__),'../../data/rnn/main/test_0828.pkl')