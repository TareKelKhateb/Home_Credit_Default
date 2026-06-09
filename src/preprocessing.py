import os
import gc
import numpy as np
import pandas as pd
from src.config import (
    BUREAU_AGG_DICT, INST_AGG_DICT, POS_AGG_DICT, CC_AGG_DICT,
    PREV_APP_NUMERICAL_FEATS, PREV_APP_DAYS_ANOMALY_COLS, PREV_APP_CAT_COLS,
    ALWAYS_KEEP_COLS, MISSING_THRESHOLD
)
from src.utils import reduce_mem_usage

def aggregate_bureau_balance(path):
    """Loads and aggregates bureau_balance.csv by SK_ID_BUREAU."""
    file_path = os.path.join(path, 'bureau_balance.csv')
    if not os.path.exists(file_path):
        print("bureau_balance.csv not found, skipping...")
        return None
        
    print("Aggregating bureau_balance.csv...")
    bb = pd.read_csv(file_path)
    bb = reduce_mem_usage(bb, verbose=False)
    
    # One-hot encoding status
    bb = pd.get_dummies(bb, columns=['STATUS'], prefix='STATUS')
    
    # Sort and take last status
    bb = bb.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])
    status_cols = [col for col in bb.columns if 'STATUS_' in col]
    last_status = bb.groupby('SK_ID_BUREAU')[status_cols].last().add_suffix('_LAST')
    
    # Main aggregation
    bb_agg_dict = {'MONTHS_BALANCE': ['max', 'min', 'size']}
    for col in status_cols:
        bb_agg_dict[col] = ['mean', 'sum']
        
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_agg_dict)
    bb_agg.columns = ['_'.join(col).strip() for col in bb_agg.columns.values]
    bb_agg = bb_agg.merge(last_status, on='SK_ID_BUREAU', how='left')
    
    del bb, last_status
    gc.collect()
    return bb_agg

def aggregate_bureau(path, df_full):
    """Loads bureau.csv, merges bureau_balance aggregates, and aggregates by SK_ID_CURR."""
    file_path = os.path.join(path, 'bureau.csv')
    if not os.path.exists(file_path):
        print("bureau.csv not found, skipping...")
        return df_full
        
    print("Aggregating bureau.csv...")
    bureau = pd.read_csv(file_path)
    bureau = reduce_mem_usage(bureau, verbose=False)
    
    # 1. Row-level feature engineering
    bureau['DEBT_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
    diff = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['CLOSED_LATE_FLAG'] = np.where(diff.isna(), np.nan, (diff > 0).astype(float))
    bureau['IS_ACTIVE_FLAG'] = bureau['DAYS_ENDDATE_FACT'].isnull().astype(int)
    bureau['SEVERE_OVERDUE_FLAG'] = (bureau['CREDIT_DAY_OVERDUE'] > 60).astype(int)
    
    # One-hot encode categoricals
    bureau = pd.get_dummies(bureau, columns=['CREDIT_TYPE', 'CREDIT_ACTIVE'])
    bureau['DAYS_CREDIT_RECENT'] = (bureau['DAYS_CREDIT'] >= -365).astype(int)
    
    # Merge bureau balance aggregates
    bb_agg = aggregate_bureau_balance(path)
    if bb_agg is not None:
        bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
        del bb_agg
        gc.collect()
        
    # Build dynamic aggregation dictionary
    local_agg_dict = BUREAU_AGG_DICT.copy()
    dynamic_cols = [col for col in bureau.columns if 'CREDIT_TYPE_' in col or 'CREDIT_ACTIVE_' in col or 'STATUS_' in col]
    for col in dynamic_cols:
        local_agg_dict[col] = ['sum', 'mean']
        
    # Group by applicant
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(local_agg_dict)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
    
    df_full = df_full.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    del bureau, bureau_agg
    gc.collect()
    return df_full

def aggregate_installments(path, df_full):
    """Loads and aggregates installments_payments.csv by SK_ID_CURR."""
    file_path = os.path.join(path, 'installments_payments.csv')
    if not os.path.exists(file_path):
        print("installments_payments.csv not found, skipping...")
        return df_full
        
    print("Aggregating installments_payments.csv...")
    inst = pd.read_csv(file_path)
    inst = reduce_mem_usage(inst, verbose=False)
    
    # Time and money gaps
    inst['PAYMENT_DPD'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
    inst['PAYMENT_DBD'] = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    
    # Group by Applicant
    inst_agg = inst.groupby('SK_ID_CURR').agg(INST_AGG_DICT)
    inst_agg.columns = ['INSTAL_' + '_'.join(col).strip() for col in inst_agg.columns.values]
    
    df_full = df_full.merge(inst_agg, on='SK_ID_CURR', how='left')
    
    del inst, inst_agg
    gc.collect()
    return df_full

def aggregate_pos_cash(path, df_full):
    """Loads and aggregates POS_CASH_balance.csv by SK_ID_CURR."""
    file_path = os.path.join(path, 'POS_CASH_balance.csv')
    if not os.path.exists(file_path):
        print("POS_CASH_balance.csv not found, skipping...")
        return df_full
        
    print("Aggregating POS_CASH_balance.csv...")
    pos = pd.read_csv(file_path)
    pos = reduce_mem_usage(pos, verbose=False)
    
    # Row-level calculations
    pos['POS_REMAINING_PROGRESS'] = pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT'].replace(0, np.nan)
    pos['POS_IS_LATE'] = (pos['SK_DPD'] > 0).astype(int)
    pos['POS_IS_LATE_WITH_TOLERANCE'] = (pos['SK_DPD_DEF'] > 0).astype(int)
    
    # One-hot encode contract status
    pos = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], prefix='POS_STAT')
    
    # Build dynamic aggregation dictionary
    local_agg_dict = POS_AGG_DICT.copy()
    status_cols = [col for col in pos.columns if 'POS_STAT_' in col]
    for col in status_cols:
        local_agg_dict[col] = ['mean', 'sum']
        
    pos_agg = pos.groupby('SK_ID_CURR').agg(local_agg_dict)
    pos_agg.columns = ['POS_' + '_'.join(col).strip() for col in pos_agg.columns.values]
    
    df_full = df_full.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    del pos, pos_agg
    gc.collect()
    return df_full

def aggregate_credit_card(path, df_full):
    """Loads and aggregates credit_card_balance.csv by SK_ID_CURR."""
    file_path = os.path.join(path, 'credit_card_balance.csv')
    if not os.path.exists(file_path):
        print("credit_card_balance.csv not found, skipping...")
        return df_full
        
    print("Aggregating credit_card_balance.csv...")
    cc = pd.read_csv(file_path)
    cc = reduce_mem_usage(cc, verbose=False)
    
    # Row-level ratios
    cc['CC_UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    cc['CC_ATM_DRAWING_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_DRAWINGS_CURRENT'].replace(0, np.nan)
    cc['CC_PAYMENT_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
    cc['CC_IS_LATE'] = (cc['SK_DPD'] > 0).astype(int)
    
    # One-hot encode status
    cc = pd.get_dummies(cc, columns=['NAME_CONTRACT_STATUS'], prefix='CC_STAT')
    
    # Build local agg dict
    local_agg_dict = CC_AGG_DICT.copy()
    status_cols = [col for col in cc.columns if 'CC_STAT_' in col]
    for col in status_cols:
        local_agg_dict[col] = ['mean', 'sum']
        
    cc_agg = cc.groupby('SK_ID_CURR').agg(local_agg_dict)
    cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
    
    df_full = df_full.merge(cc_agg, on='SK_ID_CURR', how='left')
    
    del cc, cc_agg
    gc.collect()
    return df_full

def aggregate_previous_applications(path, df_full):
    """Loads and aggregates previous_application.csv by SK_ID_CURR."""
    file_path = os.path.join(path, 'previous_application.csv')
    if not os.path.exists(file_path):
        print("previous_application.csv not found, skipping...")
        return df_full
        
    print("Aggregating previous_application.csv...")
    prev = pd.read_csv(file_path)
    prev = reduce_mem_usage(prev, verbose=False)
    
    # Row-level calculations
    prev['PREV_APP_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_APP_CREDIT_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION'].replace(0, np.nan)
    
    # Fix DAYS anomalies
    for col in PREV_APP_DAYS_ANOMALY_COLS:
        if col in prev.columns:
            prev[col] = prev[col].replace(365243, np.nan)
            
    # One-hot encode categoricals
    existing_cat = [c for c in PREV_APP_CAT_COLS if c in prev.columns]
    prev = pd.get_dummies(prev, columns=existing_cat, prefix=existing_cat)
    
    # Force numeric cast safely
    numeric_cols_to_cast = [
        'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
        'AMT_GOODS_PRICE', 'CNT_PAYMENT', 'DAYS_DECISION', 'NFLAG_INSURED_ON_APPROVAL'
    ]
    for col in numeric_cols_to_cast:
        if col in prev.columns:
            prev[col] = pd.to_numeric(prev[col], errors='coerce')
            
    # Build aggregation dictionary
    local_agg_dict = {}
    numeric_types = prev.select_dtypes(include=[np.number]).columns
    for col, aggs in PREV_APP_NUMERICAL_FEATS.items():
        if col in numeric_types and col != 'SK_ID_CURR':
            local_agg_dict[col] = aggs
            
    # Add encoded categoricals to aggregation dict
    dummy_cols = [col for col in prev.columns if any(col.startswith(f"{base}_") for base in existing_cat)]
    for col in dummy_cols:
        local_agg_dict[col] = ['mean', 'sum']
        
    needed_cols = ['SK_ID_CURR'] + list(local_agg_dict.keys())
    needed_cols = [col for col in needed_cols if col in prev.columns]
    local_agg_dict = {k: v for k, v in local_agg_dict.items() if k in prev.columns}
    
    # Group by applicant
    prev_agg = prev[needed_cols].groupby('SK_ID_CURR').agg(local_agg_dict)
    prev_agg.columns = ['PREV_' + '_'.join([str(c) for c in col]).strip() for col in prev_agg.columns.values]
    prev_agg = prev_agg.reset_index()
    
    df_full = df_full.merge(prev_agg, on='SK_ID_CURR', how='left')
    
    del prev, prev_agg
    gc.collect()
    return df_full

def build_interaction_features(df_full):
    """Computes all 8 blocks of custom engineered interaction features in place."""
    print("Computing interaction feature blocks...")
    
    # BLOCK 1: EXT_SOURCE Interactions
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df_full['EXT_SOURCE_MEAN'] = df_full[ext_cols].mean(axis=1)
    df_full['EXT_SOURCE_PROD'] = df_full[ext_cols].prod(axis=1)
    df_full['EXT_SOURCE_STD'] = df_full[ext_cols].std(axis=1)
    df_full['EXT_SOURCE_MIN'] = df_full[ext_cols].min(axis=1)
    df_full['EXT_SOURCE_MAX'] = df_full[ext_cols].max(axis=1)
    df_full['EXT_SOURCE_RANGE'] = df_full['EXT_SOURCE_MAX'] - df_full['EXT_SOURCE_MIN']
    
    df_full['EXT_1_x_EXT_2'] = df_full['EXT_SOURCE_1'] * df_full['EXT_SOURCE_2']
    df_full['EXT_1_x_EXT_3'] = df_full['EXT_SOURCE_1'] * df_full['EXT_SOURCE_3']
    df_full['EXT_2_x_EXT_3'] = df_full['EXT_SOURCE_2'] * df_full['EXT_SOURCE_3']
    df_full['EXT_SOURCE_MEAN_x_AGE'] = df_full['EXT_SOURCE_MEAN'] * (df_full['DAYS_BIRTH'] / -365)
    
    # BLOCK 2: Income / Credit / Annuity Ratios
    df_full['CREDIT_TO_INCOME_RATIO'] = df_full['AMT_CREDIT'] / df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df_full['ANNUITY_TO_INCOME_RATIO'] = df_full['AMT_ANNUITY'] / df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df_full['CREDIT_TO_GOODS_RATIO'] = df_full['AMT_CREDIT'] / df_full['AMT_GOODS_PRICE'].replace(0, np.nan)
    df_full['ANNUITY_TO_CREDIT_RATIO'] = df_full['AMT_ANNUITY'] / df_full['AMT_CREDIT'].replace(0, np.nan)
    df_full['LOAN_REPAYMENT_YEARS'] = df_full['AMT_CREDIT'] / (df_full['AMT_ANNUITY'].replace(0, np.nan) * 12)
    df_full['INCOME_PER_PERSON'] = df_full['AMT_INCOME_TOTAL'] / df_full['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df_full['INCOME_PER_CHILD'] = df_full['AMT_INCOME_TOTAL'] / (df_full['CNT_CHILDREN'].replace(0, np.nan) + 1)
    
    if 'AMT_REQ_CREDIT_BUREAU_YEAR' in df_full.columns:
        df_full['CREDIT_ENQUIRIES_TO_INCOME'] = (
            df_full['AMT_REQ_CREDIT_BUREAU_YEAR'] / df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
        )
        
    # BLOCK 3: Age & Employment Signals
    df_full['AGE_YEARS'] = df_full['DAYS_BIRTH'] / -365
    df_full['EMPLOYED_YEARS'] = df_full['DAYS_EMPLOYED'] / -365
    df_full['EMPLOYMENT_TO_AGE_RATIO'] = df_full['DAYS_EMPLOYED'] / df_full['DAYS_BIRTH'].replace(0, np.nan)
    df_full['REGISTRATION_TO_BIRTH_RATIO'] = df_full['DAYS_REGISTRATION'] / df_full['DAYS_BIRTH'].replace(0, np.nan)
    df_full['DAYS_EMPLOYED_TO_CREDIT_LENGTH'] = df_full['DAYS_EMPLOYED'] / df_full['AMT_CREDIT'].replace(0, np.nan)
    df_full['YOUNG_AND_NEW_JOB'] = ((df_full['AGE_YEARS'] < 35).astype(int) * (df_full['DAYS_EMPLOYED'] > -365).astype(int))
    
    # BLOCK 4: Bureau Debt Coverage Ratios
    if 'BUREAU_AMT_CREDIT_SUM_DEBT_sum' in df_full.columns:
        df_full['BUREAU_DEBT_TO_INCOME'] = df_full['BUREAU_AMT_CREDIT_SUM_DEBT_sum'] / df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df_full['BUREAU_DEBT_TO_CREDIT'] = df_full['BUREAU_AMT_CREDIT_SUM_DEBT_sum'] / df_full['AMT_CREDIT'].replace(0, np.nan)
        
    if 'BUREAU_AMT_CREDIT_SUM_sum' in df_full.columns:
        df_full['BUREAU_TOTAL_CREDIT_TO_INCOME'] = df_full['BUREAU_AMT_CREDIT_SUM_sum'] / df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df_full['TOTAL_DEBT_BURDEN'] = (
            (df_full.get('BUREAU_AMT_CREDIT_SUM_DEBT_sum', 0).fillna(0) + df_full['AMT_CREDIT']) /
            df_full['AMT_INCOME_TOTAL'].replace(0, np.nan)
        )
        
    if 'BUREAU_CREDIT_DAY_OVERDUE_max' in df_full.columns:
        df_full['HAS_BUREAU_OVERDUE'] = (df_full['BUREAU_CREDIT_DAY_OVERDUE_max'] > 0).astype(float)
        
    # BLOCK 5: Previous Application Signals
    refused_col = next((c for c in df_full.columns if 'NAME_CONTRACT_STATUS_Refused' in c and 'PREV_' in c), None)
    approved_col = next((c for c in df_full.columns if 'NAME_CONTRACT_STATUS_Approved' in c and 'PREV_' in c), None)
    total_prev_col = next((c for c in df_full.columns if 'PREV_SK_ID_PREV_count' in c), None)
    
    if refused_col and total_prev_col:
        df_full['PREV_REJECTION_RATE'] = df_full[refused_col] / df_full[total_prev_col].replace(0, np.nan)
    if approved_col and total_prev_col:
        df_full['PREV_APPROVAL_RATE'] = df_full[approved_col] / df_full[total_prev_col].replace(0, np.nan)
        
    if 'PREV_AMT_APPLICATION_mean' in df_full.columns and 'PREV_AMT_CREDIT_mean' in df_full.columns:
        df_full['PREV_CREDIT_TO_APPLICATION_RATIO'] = (
            df_full['PREV_AMT_CREDIT_mean'] / df_full['PREV_AMT_APPLICATION_mean'].replace(0, np.nan)
        )
        
    # BLOCK 6: Installment Payment Behavior Ratios
    if 'INSTAL_PAYMENT_DPD_mean' in df_full.columns and 'INSTAL_AMT_INSTALMENT_mean' in df_full.columns:
        df_full['INSTAL_DPD_PER_PAYMENT'] = df_full['INSTAL_PAYMENT_DPD_mean'] / df_full['INSTAL_AMT_INSTALMENT_mean'].replace(0, np.nan)
        
    if 'INSTAL_PAYMENT_DIFF_mean' in df_full.columns and 'INSTAL_AMT_INSTALMENT_mean' in df_full.columns:
        df_full['INSTAL_UNDERPAYMENT_RATIO'] = df_full['INSTAL_PAYMENT_DIFF_mean'] / df_full['INSTAL_AMT_INSTALMENT_mean'].replace(0, np.nan)
        
    if 'INSTAL_AMT_PAYMENT_sum' in df_full.columns and 'INSTAL_AMT_INSTALMENT_sum' in df_full.columns:
        df_full['INSTAL_PAYMENT_COMPLETION_RATIO'] = df_full['INSTAL_AMT_PAYMENT_sum'] / df_full['INSTAL_AMT_INSTALMENT_sum'].replace(0, np.nan)
        
    # BLOCK 7: Credit Card Behavioral Signals
    if 'CC_CC_UTILIZATION_mean' in df_full.columns:
        df_full['CC_HIGH_UTILIZATION_FLAG'] = (df_full['CC_CC_UTILIZATION_mean'] > 0.8).astype(float)
    if 'CC_CC_ATM_DRAWING_RATIO_mean' in df_full.columns:
        df_full['CC_CASH_DEPENDENT_FLAG'] = (df_full['CC_CC_ATM_DRAWING_RATIO_mean'] > 0.5).astype(float)
    if 'CC_SK_DPD_max' in df_full.columns and 'CC_MONTHS_BALANCE_size' in df_full.columns:
        df_full['CC_DPD_PER_MONTH'] = df_full['CC_CC_IS_LATE_sum'] / df_full['CC_MONTHS_BALANCE_size'].replace(0, np.nan)
        
    # BLOCK 8: Document / Social Completeness Scores
    doc_cols = [col for col in df_full.columns if col.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df_full['TOTAL_DOCS_PROVIDED'] = df_full[doc_cols].sum(axis=1)
        
    social_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    existing_social = [c for c in social_cols if c in df_full.columns]
    if existing_social:
        df_full['TOTAL_SOCIAL_CONTACTS'] = df_full[existing_social].sum(axis=1)
        
    if 'REGION_RATING_CLIENT' in df_full.columns and 'REGION_RATING_CLIENT_W_CITY' in df_full.columns:
        df_full['REGION_RATING_MISMATCH'] = (df_full['REGION_RATING_CLIENT'] != df_full['REGION_RATING_CLIENT_W_CITY']).astype(int)
        
    return df_full

def preprocess_data(data_path, df_full):
    """
    Main preprocessing pipeline orchestrator.
    Fuses supplemental relational tables, performs aggregations and engineering,
    and drops columns with high missing ratios.
    """
    df_full = aggregate_bureau(data_path, df_full)
    df_full = aggregate_installments(data_path, df_full)
    df_full = aggregate_pos_cash(data_path, df_full)
    df_full = aggregate_credit_card(data_path, df_full)
    df_full = aggregate_previous_applications(data_path, df_full)
    
    # Engineered interaction features
    df_full = build_interaction_features(df_full)
    
    # Drop columns with missing percent > threshold (excluding always keeps)
    missing_pct = df_full.isnull().mean()
    high_missing = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    high_missing = [col for col in high_missing if col not in ALWAYS_KEEP_COLS]
    
    print(f"Dropping {len(high_missing)} columns with >{MISSING_THRESHOLD*100}% missing values...")
    df_full.drop(columns=high_missing, inplace=True, errors='ignore')
    
    df_full = reduce_mem_usage(df_full)
    gc.collect()
    
    return df_full
