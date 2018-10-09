from enum import Enum
import sys,os

class FILE(Enum):
    train_ori = os.path.join(os.path.dirname(__file__),'../../data/original/train.pkl')
    test_ori = os.path.join(os.path.dirname(__file__),'../../data/original/test.pkl')
    
    train_final = os.path.join(os.path.dirname(__file__),'../../data/original/train_ori_comb_hour.pkl')
    test_final = os.path.join(os.path.dirname(__file__),'../../data/original/test_ori_r2_hour.pkl')

    