from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CategoricalNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Cleans categorical features by:
    1. Identifying features containing noise strings like 'XNA' or 'Unknown'.
    2. Replacing those noise strings with standard 'NaN'.
    3. Filling actual missing values with the string 'NaN' so that they are treated as a distinct group.
    """
    def __init__(self):
        self.all_cat_cols_ = []
        self.cols_with_noise_ = []

    def fit(self, X, y=None):
        self.all_cat_cols_ = X.select_dtypes(include=['object', 'category', 'str', 'string']).columns.tolist()
        self.cols_with_noise_ = [
            col for col in self.all_cat_cols_
            if X[col].astype(str).isin(['XNA', 'Unknown']).any()
        ]
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in self.all_cat_cols_:
            if col not in X_transformed.columns:
                continue  # Skip if the column was dropped
            X_transformed[col] = X_transformed[col].astype(str)
            if col in self.cols_with_noise_:
                X_transformed[col] = X_transformed[col].replace(['XNA', 'Unknown'], 'NaN')
            X_transformed[col] = X_transformed[col].fillna('NaN')
        return X_transformed

class DaysEmployedAnomalyTransformer(BaseEstimator, TransformerMixin):
    """
    Handles the DAYS_EMPLOYED anomaly in the Home Credit dataset where the value 365243 
    is used as a placeholder (typically representing retirees/unemployed).
    It replaces 365243 with np.nan and creates a binary indicator column 'DAYS_EMPLOYED_ANOM'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        if 'DAYS_EMPLOYED' in X_transformed.columns:
            X_transformed['DAYS_EMPLOYED_ANOM'] = (X_transformed['DAYS_EMPLOYED'] == 365243).astype(int)
            X_transformed['DAYS_EMPLOYED'] = X_transformed['DAYS_EMPLOYED'].replace(365243, np.nan)
        return X_transformed
