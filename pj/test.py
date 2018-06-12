import unittest
from Main import Feature
import pandas as pd

class Test(unittest.TestCase):
    
    def test_count(self):
        df = pd.DataFrame({'a':[0,0,0,0,1,1,1,1], 
                           'b':[0,1,0,1,1,1,1,0],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))        
        returned = Feature.count(df, cols=['a','b'], col_name='count_a_b', params={'col':'label'})['count_a_b'].values
        self.assertEqual(list(returned), [2, 2, 2, 2, 3, 3, 3, 1])

    def test_unique_count(self):
        df = pd.DataFrame({'a':[0,0,0,0,1,1,1,1], 
                           'b':[0,1,2,3,1,1,1,1],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.unique_count(df, cols=['a','b'], col_name='unique_count_a_b', params=None)['unique_count_a_b'].values
        self.assertEqual(list(returned), [4,4,4,4,1,1,1,1]) 

    def test_cumulative_count(self):
        df = pd.DataFrame({'a':[0,0,1,0,1,2,0,1], 
                           'b':[0,1,2,3,1,1,1,1],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.cumulative_count(df, cols=['a'], col_name='cum_count_a_b', params=None)['cum_count_a_b'].values
        self.assertEqual(list(returned), [0,1,0,2,1,0,3,2])
        
    def test_reverse_cumulative_count(self):
        df = pd.DataFrame({'a':[0,0,1,0,1,2,0,1], 
                           'b':[0,1,2,3,1,1,1,1],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.reverse_cumulative_count(df, cols=['a'], col_name='rev_cum_count_a_b', params=None)['rev_cum_count_a_b'].values
        self.assertEqual(list(returned), [3,2,2,1,1,0,0,0])
        
    def test_time_to_n_next(self):
        df = pd.DataFrame({'a':[0,1,1,1,2,0,1,0], 
                           't':[0,1,2,3,4,5,6,7],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.time_to_n_next(df, cols=['a','t'], col_name='time_to_n_next', params={'n':'1', 'fillna': '222'}).values
        self.assertEqual(list(returned), [5,1,1,3,222,2,222,222])
        
    def test_count_in_previous_n_time_unit(self):
        df = pd.DataFrame({'a':[0,1,1,1,1,0,1,0], 
                           't':[0,1,2,3,4,5,6,7],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.count_in_previous_n_time_unit(df, cols=['a','t'], col_name='count_prev_n', params={'n':'3'}).values
        self.assertEqual(list(returned), [0,0,1,2,3,0,2,1])    
        
    def test_count_in_next_n_time_unit(self):
        df = pd.DataFrame({'a':[0,1,1,1,1,0,1,0], 
                           't':[0,1,2,3,4,5,6,7],
                           'label':[1,1,1,0,1,0,1,0]},
                           index=range(8))
        returned = Feature.count_in_next_n_time_unit(df, cols=['a','t'], col_name='count_next_n', params={'n':'3'}).values
        self.assertEqual(list(returned), [0,3,2,2,1,1,0,0])       
        
if __name__ == '__main__':
    unittest.main()