import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reduce_mem_usage(df, verbose=True):
    """
    Downcasts numerical columns in a pandas DataFrame to optimize memory footprint.
    Decreases memory usage by up to 50%+.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            col_type = df[col].dtype
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage decreased to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def plot_feature_importances(importance_df, top_n=25, output_path=None):
    """
    Plots the top N features and optionally saves the image to a file.
    """
    top_features = importance_df.head(top_n)
    plt.figure(figsize=(10, 10))
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved feature importances plot to {output_path}")
    plt.close()

def generate_mock_data(output_dir='data', sample_size=100):
    """
    Generates synthetic dummy CSV files representing all relational inputs.
    Enables quick training dry-runs without downloading massive Kaggle sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic mock data in '{output_dir}' with base sample size {sample_size}...")

    # Set seed for reproducibility
    np.random.seed(42)

    # 1. Generate applicant IDs
    curr_ids = list(range(100001, 100001 + sample_size))
    test_ids = list(range(200001, 200001 + int(sample_size * 0.3)))

    # Helper helper
    def create_application_df(ids, is_train=True):
        n = len(ids)
        cnt_children = np.random.randint(0, 4, size=n)
        data = {
            'SK_ID_CURR': ids,
            'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n, p=[0.9, 0.1]),
            'CODE_GENDER': np.random.choice(['M', 'F', 'XNA'], size=n, p=[0.35, 0.649, 0.001]),
            'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], size=n),
            'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], size=n),
            'CNT_CHILDREN': cnt_children,
            'CNT_FAM_MEMBERS': cnt_children + np.random.choice([1, 2], size=n),
            'DAYS_REGISTRATION': np.random.randint(-15000, -10, size=n),
            'AMT_INCOME_TOTAL': np.random.exponential(scale=150000, size=n) + 20000,
            'AMT_CREDIT': np.random.exponential(scale=500000, size=n) + 50000,
            'AMT_ANNUITY': np.random.exponential(scale=25000, size=n) + 2000,
            'AMT_GOODS_PRICE': np.random.exponential(scale=450000, size=n) + 40000,
            'NAME_TYPE_SUITE': np.random.choice(['Unaccompanied', 'Family', 'Spouse, partner', 'NaN'], size=n),
            'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant'], size=n),
            'NAME_EDUCATION_TYPE': np.random.choice(['Secondary / secondary special', 'Higher education', 'Incomplete higher'], size=n),
            'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single / not married', 'Civil marriage'], size=n),
            'NAME_HOUSING_TYPE': np.random.choice(['House / apartment', 'With parents', 'Rented apartment'], size=n),
            'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Sales staff', 'Managers', 'NaN'], size=n),
            'WEEKDAY_APPR_PROCESS_START': np.random.choice(['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY'], size=n),
            'ORGANIZATION_TYPE': np.random.choice(['Business Entity Type 3', 'XNA', 'Self-employed', 'Other'], size=n),
            'FONDKAPREMONT_MODE': np.random.choice(['reg oper account', 'NaN'], size=n),
            'HOUSETYPE_MODE': np.random.choice(['block of flats', 'NaN'], size=n),
            'WALLSMATERIAL_MODE': np.random.choice(['Panel', 'Stone, brick', 'NaN'], size=n),
            'EMERGENCYSTATE_MODE': np.random.choice(['No', 'Yes', 'NaN'], size=n),
            'DAYS_BIRTH': np.random.randint(-25000, -7000, size=n),
            # Add DAYS_EMPLOYED with the anomaly placeholder (365243)
            'DAYS_EMPLOYED': np.random.choice([365243] + list(np.random.randint(-15000, -100, size=19)), size=n),
            'EXT_SOURCE_1': np.random.uniform(0.1, 0.9, size=n),
            'EXT_SOURCE_2': np.random.uniform(0.1, 0.9, size=n),
            'EXT_SOURCE_3': np.random.uniform(0.1, 0.9, size=n),
            'REGION_POPULATION_RELATIVE': np.random.uniform(0.001, 0.05, size=n),
            'DAYS_ID_PUBLISH': np.random.randint(-5000, -10, size=n)
        }
        # Add FLAG_DOCUMENT flags
        for i in range(2, 22):
            data[f'FLAG_DOCUMENT_{i}'] = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
            
        if is_train:
            data['TARGET'] = np.random.choice([0, 1], size=n, p=[0.92, 0.08])
            
        return pd.DataFrame(data)

    df_train = create_application_df(curr_ids, is_train=True)
    df_test = create_application_df(test_ids, is_train=False)

    df_train.to_csv(os.path.join(output_dir, 'application_train.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'application_test.csv'), index=False)

    # 2. Bureau and Bureau Balance
    bureau_data = []
    bureau_balance_data = []
    bureau_id = 5000001

    all_ids = curr_ids + test_ids
    for curr_id in all_ids:
        # Generate 0 to 4 credits per applicant
        num_credits = np.random.randint(0, 5)
        for _ in range(num_credits):
            bureau_data.append({
                'SK_ID_CURR': curr_id,
                'SK_ID_BUREAU': bureau_id,
                'CREDIT_ACTIVE': np.random.choice(['Closed', 'Active'], p=[0.6, 0.4]),
                'CREDIT_TYPE': np.random.choice(['Consumer credit', 'Credit card', 'Car loan'], p=[0.7, 0.2, 0.1]),
                'DAYS_CREDIT': np.random.randint(-2900, -10),
                'CREDIT_DAY_OVERDUE': np.random.choice([0, np.random.randint(1, 100)], p=[0.95, 0.05]),
                'DAYS_CREDIT_ENDDATE': np.random.randint(-1000, 1000),
                'DAYS_ENDDATE_FACT': np.random.choice([np.nan, np.random.randint(-1500, -10)], p=[0.4, 0.6]),
                'AMT_CREDIT_MAX_OVERDUE': np.random.exponential(scale=5000) * np.random.choice([0, 1], p=[0.8, 0.2]),
                'CNT_CREDIT_PROLONG': np.random.choice([0, 1, 2], p=[0.98, 0.015, 0.005]),
                'AMT_CREDIT_SUM': np.random.exponential(scale=100000) + 10000,
                'AMT_CREDIT_SUM_DEBT': np.random.exponential(scale=50000) * np.random.choice([0, 1], p=[0.5, 0.5]),
                'AMT_CREDIT_SUM_LIMIT': np.random.exponential(scale=10000) * np.random.choice([0, 1], p=[0.9, 0.1]),
                'DAYS_CREDIT_UPDATE': np.random.randint(-1000, -1)
            })

            # Generate monthly balance for this bureau credit
            months = np.random.randint(5, 30)
            for m in range(months):
                bureau_balance_data.append({
                    'SK_ID_BUREAU': bureau_id,
                    'MONTHS_BALANCE': -m,
                    'STATUS': np.random.choice(['C', '0', '1', 'X', '2'], p=[0.5, 0.4, 0.05, 0.045, 0.005])
                })
            bureau_id += 1

    pd.DataFrame(bureau_data).to_csv(os.path.join(output_dir, 'bureau.csv'), index=False)
    pd.DataFrame(bureau_balance_data).to_csv(os.path.join(output_dir, 'bureau_balance.csv'), index=False)

    # 3. Previous Application
    prev_data = []
    inst_data = []
    pos_data = []
    cc_data = []
    prev_id = 1000001

    for curr_id in all_ids:
        num_prev = np.random.randint(0, 4)
        for _ in range(num_prev):
            status = np.random.choice(['Approved', 'Refused', 'Canceled', 'Unused offer'], p=[0.6, 0.2, 0.15, 0.05])
            prev_data.append({
                'SK_ID_PREV': prev_id,
                'SK_ID_CURR': curr_id,
                'NAME_CONTRACT_STATUS': status,
                'AMT_ANNUITY': np.random.exponential(scale=10000) + 1000,
                'AMT_APPLICATION': np.random.exponential(scale=150000) + 5000,
                'AMT_CREDIT': np.random.exponential(scale=150000) + 5000,
                'AMT_DOWN_PAYMENT': np.random.exponential(scale=10000) * np.random.choice([0, 1], p=[0.7, 0.3]),
                'AMT_GOODS_PRICE': np.random.exponential(scale=140000) + 5000,
                'DAYS_DECISION': np.random.randint(-2900, -10),
                'CNT_PAYMENT': np.random.choice([6, 12, 18, 24, 36], p=[0.1, 0.4, 0.2, 0.2, 0.1]),
                'NFLAG_INSURED_ON_APPROVAL': np.random.choice([0.0, 1.0], p=[0.7, 0.3]),
                'NAME_YIELD_GROUP': np.random.choice(['low_normal', 'middle', 'high', 'XNA']),
                'CODE_REJECT_REASON': np.random.choice(['XAP', 'HC', 'LIMIT', 'XNA']),
                'NAME_PORTFOLIO': np.random.choice(['POS', 'Cash', 'Cards', 'XNA']),
                'NAME_PRODUCT_TYPE': np.random.choice(['XNA', 'walk-in', 'x-sell']),
                'DAYS_FIRST_DRAWING': np.random.choice([365243.0, np.random.randint(-1000, -10)]),
                'DAYS_FIRST_DUE': np.random.choice([365243.0, np.random.randint(-1000, -10)]),
                'DAYS_LAST_DUE_1ST_VERSION': np.random.choice([365243.0, np.random.randint(-1000, -10)]),
                'DAYS_LAST_DUE': np.random.choice([365243.0, np.random.randint(-1000, -10)]),
                'DAYS_TERMINATION': np.random.choice([365243.0, np.random.randint(-1000, -10)])
            })

            # If approved, generate payment and snapshot histories
            if status == 'Approved':
                # Generate installments
                num_inst = np.random.randint(3, 12)
                for inst_idx in range(1, num_inst + 1):
                    days_inst = -30 * inst_idx
                    days_entry = days_inst + np.random.randint(-5, 5)
                    amt_inst = np.random.exponential(scale=8000) + 1000
                    amt_pay = amt_inst - (np.random.exponential(scale=500) * np.random.choice([0, 1], p=[0.9, 0.1]))
                    inst_data.append({
                        'SK_ID_PREV': prev_id,
                        'SK_ID_CURR': curr_id,
                        'NUM_INSTALMENT_VERSION': 1.0,
                        'NUM_INSTALMENT_NUMBER': inst_idx,
                        'DAYS_INSTALMENT': days_inst,
                        'DAYS_ENTRY_PAYMENT': days_entry,
                        'AMT_INSTALMENT': amt_inst,
                        'AMT_PAYMENT': amt_pay
                    })

                # Generate POS CASH balance snapshots
                pos_months = np.random.randint(2, 10)
                for pm in range(pos_months):
                    pos_data.append({
                        'SK_ID_PREV': prev_id,
                        'SK_ID_CURR': curr_id,
                        'MONTHS_BALANCE': -pm,
                        'CNT_INSTALMENT': 12.0,
                        'CNT_INSTALMENT_FUTURE': max(0.0, 12.0 - pm),
                        'NAME_CONTRACT_STATUS': 'Active',
                        'SK_DPD': np.random.choice([0, np.random.randint(1, 10)], p=[0.98, 0.02]),
                        'SK_DPD_DEF': np.random.choice([0, np.random.randint(1, 10)], p=[0.99, 0.01])
                    })

                # Generate credit card balances for some Card previous accounts
                if np.random.choice([True, False], p=[0.15, 0.85]):
                    cc_months = np.random.randint(2, 10)
                    for ccm in range(cc_months):
                        balance = np.random.exponential(scale=20000)
                        limit = 50000.0
                        draw_atm = np.random.exponential(scale=5000) * np.random.choice([0, 1], p=[0.8, 0.2])
                        draw_current = draw_atm + np.random.exponential(scale=2000)
                        cc_data.append({
                            'SK_ID_PREV': prev_id,
                            'SK_ID_CURR': curr_id,
                            'MONTHS_BALANCE': -ccm,
                            'AMT_BALANCE': balance,
                            'AMT_CREDIT_LIMIT_ACTUAL': limit,
                            'AMT_DRAWINGS_ATM_CURRENT': draw_atm,
                            'AMT_DRAWINGS_CURRENT': draw_current,
                            'CNT_DRAWINGS_ATM_CURRENT': int(draw_atm > 0),
                            'CNT_DRAWINGS_CURRENT': int(draw_current > 0) + 1,
                            'SK_DPD': np.random.choice([0, np.random.randint(1, 10)], p=[0.98, 0.02]),
                            'SK_DPD_DEF': np.random.choice([0, np.random.randint(1, 10)], p=[0.99, 0.01]),
                            'NAME_CONTRACT_STATUS': 'Active',
                            'AMT_PAYMENT_CURRENT': balance * 0.1,
                            'AMT_INST_MIN_REGULARITY': balance * 0.05
                        })
            prev_id += 1

    pd.DataFrame(prev_data).to_csv(os.path.join(output_dir, 'previous_application.csv'), index=False)
    pd.DataFrame(inst_data).to_csv(os.path.join(output_dir, 'installments_payments.csv'), index=False)
    pd.DataFrame(pos_data).to_csv(os.path.join(output_dir, 'POS_CASH_balance.csv'), index=False)
    pd.DataFrame(cc_data).to_csv(os.path.join(output_dir, 'credit_card_balance.csv'), index=False)

    # 4. Generate column descriptions mock CSV
    cols_desc = [
        {'Row': 'AMT_CREDIT_MAX_OVERDUE', 'Description': 'Maximal amount overdue on the Credit Bureau credit so far'},
        {'Row': 'CNT_CREDIT_PROLONG', 'Description': 'How many times was the Credit Bureau credit prolonged'},
        {'Row': 'DAYS_EMPLOYED', 'Description': 'How many days before the application the person started current job'}
    ]
    pd.DataFrame(cols_desc).to_csv(os.path.join(output_dir, 'HomeCredit_columns_description.csv'), index=False)

    print("Mock data generation completed successfully!")
