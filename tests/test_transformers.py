import unittest
import pandas as pd
import numpy as np
from src.transformers import CategoricalNaNTransformer, DaysEmployedAnomalyTransformer

class TestTransformers(unittest.TestCase):
    
    def test_categorical_nan_transformer(self):
        # Create dummy df with categorical noise and nulls
        df = pd.DataFrame({
            'GENDER': ['M', 'F', 'XNA', None, 'F'],
            'INCOME_TYPE': ['Working', 'Unknown', 'Pensioner', 'Working', 'Working'],
            'NUMERIC_COL': [1, 2, 3, 4, 5]
        })
        
        transformer = CategoricalNaNTransformer()
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Verify columns identified
        self.assertIn('GENDER', transformer.all_cat_cols_)
        self.assertIn('INCOME_TYPE', transformer.all_cat_cols_)
        self.assertNotIn('NUMERIC_COL', transformer.all_cat_cols_)
        
        # Verify noise columns identified
        self.assertIn('GENDER', transformer.cols_with_noise_)
        self.assertIn('INCOME_TYPE', transformer.cols_with_noise_)
        
        # Verify noise strings and NaNs replaced with 'NaN'
        self.assertEqual(df_transformed['GENDER'].iloc[2], 'NaN')  # XNA replaced
        self.assertEqual(df_transformed['GENDER'].iloc[3], 'NaN')  # None replaced
        self.assertEqual(df_transformed['INCOME_TYPE'].iloc[1], 'NaN')  # Unknown replaced
        
        # Verify non-noise values kept intact
        self.assertEqual(df_transformed['GENDER'].iloc[0], 'M')
        self.assertEqual(df_transformed['INCOME_TYPE'].iloc[0], 'Working')
        self.assertEqual(df_transformed['NUMERIC_COL'].iloc[0], 1)

    def test_days_employed_anomaly_transformer(self):
        # Create dummy df with DAYS_EMPLOYED containing the anomaly value
        df = pd.DataFrame({
            'DAYS_EMPLOYED': [-100, -200, 365243, -500],
            'OTHER_COL': [1, 2, 3, 4]
        })
        
        transformer = DaysEmployedAnomalyTransformer()
        transformer.fit(df)
        df_transformed = transformer.transform(df)
        
        # Verify DAYS_EMPLOYED_ANOM column created
        self.assertIn('DAYS_EMPLOYED_ANOM', df_transformed.columns)
        
        # Verify values in indicator column
        self.assertEqual(df_transformed['DAYS_EMPLOYED_ANOM'].iloc[0], 0)
        self.assertEqual(df_transformed['DAYS_EMPLOYED_ANOM'].iloc[2], 1)
        
        # Verify anomaly replaced with nan
        self.assertTrue(np.isnan(df_transformed['DAYS_EMPLOYED'].iloc[2]))
        
        # Verify normal values kept
        self.assertEqual(df_transformed['DAYS_EMPLOYED'].iloc[0], -100)

if __name__ == '__main__':
    unittest.main()
