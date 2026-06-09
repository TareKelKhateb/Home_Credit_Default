import numpy as np

# Global Project Settings
TARGET = 'TARGET'
RANDOM_STATE = 42

# Memory & Preprocessing Settings
ALWAYS_KEEP_COLS = [
    'TARGET', 'SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
]
MISSING_THRESHOLD = 0.65  # Drop columns missing more than 65%

# Aggregation Configuration Dictionaries
BUREAU_AGG_DICT = {
    'DAYS_CREDIT': ['min', 'max', 'mean', 'count'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['mean', 'max'],
    'DAYS_CREDIT_RECENT': ['sum'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'SEVERE_OVERDUE_FLAG': ['sum'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
    'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
    'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean'],
    'DEBT_CREDIT_RATIO': ['mean', 'max'],
    'CNT_CREDIT_PROLONG': ['sum', 'max'],
    'CLOSED_LATE_FLAG': ['sum', 'mean'],
    'IS_ACTIVE_FLAG': ['sum', 'mean']
}

INST_AGG_DICT = {
    'PAYMENT_DPD': ['max', 'mean', 'sum'],
    'PAYMENT_DBD': ['max', 'mean', 'sum'],
    'PAYMENT_DIFF': ['max', 'mean', 'sum'],
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['size']  # Count of installments
}

POS_AGG_DICT = {
    'MONTHS_BALANCE': ['max', 'min', 'size'],
    'CNT_INSTALMENT': ['max', 'mean'],
    'CNT_INSTALMENT_FUTURE': ['max', 'mean', 'sum'],
    'SK_DPD': ['max', 'mean'],
    'SK_DPD_DEF': ['max', 'mean'],
    'POS_REMAINING_PROGRESS': ['max', 'mean'],
    'POS_IS_LATE': ['sum', 'mean'],
    'POS_IS_LATE_WITH_TOLERANCE': ['sum', 'mean']
}

CC_AGG_DICT = {
    'MONTHS_BALANCE': ['max', 'min', 'size'],
    'AMT_BALANCE': ['max', 'mean', 'sum'],
    'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
    'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
    'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
    'CNT_DRAWINGS_CURRENT': ['max', 'sum'],
    'SK_DPD': ['max', 'mean'],
    'CC_UTILIZATION': ['max', 'mean'],
    'CC_ATM_DRAWING_RATIO': ['max', 'mean'],
    'CC_PAYMENT_RATIO': ['max', 'mean'],
    'CC_IS_LATE': ['sum', 'mean']
}

PREV_APP_NUMERICAL_FEATS = {
    'SK_ID_PREV': ['count'],
    'AMT_ANNUITY': ['max', 'mean', 'sum'],
    'AMT_APPLICATION': ['max', 'mean', 'sum'],
    'AMT_CREDIT': ['max', 'mean', 'sum'],
    'AMT_DOWN_PAYMENT': ['max', 'mean'],
    'AMT_GOODS_PRICE': ['max', 'mean', 'sum'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['sum', 'mean'],
    'PREV_APP_CREDIT_DIFF': ['max', 'mean', 'sum'],
    'PREV_APP_CREDIT_RATIO': ['min', 'max', 'mean'],
    'NFLAG_INSURED_ON_APPROVAL': ['sum']
}

PREV_APP_DAYS_ANOMALY_COLS = [
    'DAYS_FIRST_DRAWING',
    'DAYS_FIRST_DUE',
    'DAYS_LAST_DUE_1ST_VERSION',
    'DAYS_LAST_DUE',
    'DAYS_TERMINATION'
]

PREV_APP_CAT_COLS = [
    'NAME_CONTRACT_STATUS',
    'NAME_YIELD_GROUP',
    'CODE_REJECT_REASON',
    'NAME_PORTFOLIO',
    'NAME_PRODUCT_TYPE'
]

# Model Parameters (Default GPU Configuration)
CATBOOST_PARAMS = {
    'random_state': RANDOM_STATE,
    'iterations': 3000,
    'verbose': 100,
    'task_type': 'GPU',
    'devices': '0',
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',
    'early_stopping_rounds': 100
}
