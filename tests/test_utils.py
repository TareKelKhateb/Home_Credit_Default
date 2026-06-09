import unittest
import pandas as pd
import numpy as np
from src.utils import reduce_mem_usage

class TestUtils(unittest.TestCase):
    
    def test_reduce_mem_usage_int(self):
        # Create a dataframe with various numeric ranges
        df = pd.DataFrame({
            'small_int': [0, 5, 10],            # fits in int8
            'medium_int': [-1000, 0, 1000],      # fits in int16
            'large_int': [-50000, 0, 50000],    # fits in int32
            'huge_int': [-3000000000, 0, 3000000000] # fits in int64
        })
        
        # Keep track of original datatypes
        self.assertEqual(df['small_int'].dtype, np.int64)
        self.assertEqual(df['medium_int'].dtype, np.int64)
        
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)
        
        # Verify downcasting occurred appropriately
        self.assertEqual(df_reduced['small_int'].dtype, np.int8)
        self.assertEqual(df_reduced['medium_int'].dtype, np.int16)
        self.assertEqual(df_reduced['large_int'].dtype, np.int32)
        self.assertEqual(df_reduced['huge_int'].dtype, np.int64)

    def test_reduce_mem_usage_float(self):
        # Create a dataframe with float cols
        df = pd.DataFrame({
            'small_float': [1.1, 2.2, 3.3],            # fits in float16
            'large_float': [1e10, -1e10, 2e10]         # fits in float32 or float64 depending on range
        })
        
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)
        
        # Verify float reduction
        self.assertTrue(df_reduced['small_float'].dtype in [np.float16, np.float32])
        self.assertTrue(df_reduced['large_float'].dtype in [np.float32, np.float64])

    def test_reduce_mem_usage_non_numeric(self):
        # Non-numeric columns should not be modified
        df = pd.DataFrame({
            'string_col': ['a', 'b', 'c'],
            'categorical_col': pd.Categorical(['x', 'y', 'z'])
        })
        
        df_reduced = reduce_mem_usage(df.copy(), verbose=False)
        
        self.assertEqual(df_reduced['string_col'].dtype, df['string_col'].dtype)
        self.assertTrue(isinstance(df_reduced['categorical_col'].dtype, pd.CategoricalDtype))

if __name__ == '__main__':
    unittest.main()
