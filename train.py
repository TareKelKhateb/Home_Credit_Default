import os
import argparse
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import kagglehub
from src.config import TARGET, RANDOM_STATE
from src.preprocessing import preprocess_data
from src.pipeline import create_model_pipeline
from src.utils import reduce_mem_usage, plot_feature_importances, generate_mock_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost Pipeline for Home Credit Default Risk")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the CSV datasets')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save trained models')
    parser.add_argument('--dry-run', action='store_true', help='Generate mock data and run a fast dry-run training')
    parser.add_argument('--sample-rows', type=int, default=None, help='Load only the first N rows of the dataset for testing')
    parser.add_argument('--no-gpu', action='store_true', help='Force training on CPU instead of GPU')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 1. Handle Data Sources (Mock or Kaggle Download)
    data_path = args.data_dir
    if args.dry_run:
        print("--- RUNNING IN DRY RUN MODE ---")
        generate_mock_data(output_dir=data_path, sample_size=1000)
    else:
        # Check if main train file exists. If not, trigger kagglehub download
        train_file = os.path.join(data_path, 'application_train.csv')
        if not os.path.exists(train_file):
            print("application_train.csv not found locally. Downloading from Kaggle via kagglehub...")
            try:
                data_path = kagglehub.competition_download('home-credit-default-risk')
                print(f"Dataset downloaded successfully to cache path: {data_path}")
            except Exception as e:
                print(f"Kaggle download failed: {e}")
                print("Generating mock data fallback so training doesn't break...")
                generate_mock_data(output_dir=data_path, sample_size=1000)
        else:
            print(f"Loading data from local directory: {data_path}")
            
    # 2. Load application_train and application_test
    print("\n--- LOADING APPLICATION DATA ---")
    train_csv = os.path.join(data_path, 'application_train.csv')
    test_csv = os.path.join(data_path, 'application_test.csv')
    
    if args.sample_rows:
        print(f"Sampling mode enabled: loading first {args.sample_rows} rows.")
        df_train = pd.read_csv(train_csv, nrows=args.sample_rows)
        df_test = pd.read_csv(test_csv, nrows=args.sample_rows)
    else:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        
    print(f"Original Train shape: {df_train.shape}")
    print(f"Original Test shape: {df_test.shape}")
    
    # Concatenate to apply uniform aggregation/preprocessing
    df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_full = reduce_mem_usage(df_full, verbose=True)
    
    # Free memory
    del df_train, df_test
    gc.collect()
    
    # 3. Aggregation and feature engineering
    print("\n--- PREPROCESSING AND MERGING DATA ---")
    df_full = preprocess_data(data_path, df_full)
    
    # Split back into train and test
    train_df = df_full[df_full[TARGET].notnull()].copy()
    test_df = df_full[df_full[TARGET].isnull()].copy()
    
    print(f"Processed Train set shape: {train_df.shape}")
    print(f"Processed Test set shape: {test_df.shape}")
    
    # Save the processed test set features for predict.py later
    os.makedirs(os.path.join(args.model_dir, 'temp'), exist_ok=True)
    test_features_path = os.path.join(args.model_dir, 'temp', 'processed_test_features.joblib')
    joblib.dump(test_df, test_features_path)
    print(f"Saved preprocessed test features to cache for inference script.")
    
    # 4. Prepare training splits
    X = train_df.drop(columns=[TARGET, 'SK_ID_CURR'])
    y = train_df[TARGET]
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    
    # Free memory
    del df_full, train_df, test_df
    gc.collect()
    
    # Identify final feature types
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'str', 'string']).columns.tolist()
    print(f"\nFinal feature selection:")
    print(f"  Numerical features: {len(numerical_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # 5. Build Pipeline
    print("\n--- BUILDING SKLEARN PIPELINE ---")
    custom_params = {}
    if args.no_gpu:
        custom_params['task_type'] = 'CPU'
        
    model_pipeline = create_model_pipeline(categorical_features, custom_params)
    
    # Pre-transform test set inputs for the classifier validation step
    print("Pre-transforming test split for validation...")
    X_test_transformed = model_pipeline[:-1].fit_transform(X_test)
    
    # 6. Fit Pipeline
    print("Fitting model...")
    model_pipeline.fit(
        X_train, 
        y_train, 
        classifier__eval_set=(X_test_transformed, y_test)
    )
    
    # 7. Evaluate and plot importances
    print("\n--- MODEL EVALUATION ---")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score on Test Split: {auc_score:.4f}")
    
    # Extract Feature Importances
    classifier_model = model_pipeline.named_steps['classifier']
    feature_names = classifier_model.feature_names_
    feature_importances = classifier_model.get_feature_importance()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 15 Feature Importances:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save importance plot
    importance_plot_path = os.path.join(args.model_dir, 'feature_importances.png')
    plot_feature_importances(importance_df, top_n=25, output_path=importance_plot_path)
    
    # 8. Save Pipeline Model
    model_path = os.path.join(args.model_dir, 'catboost_pipeline.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"\nSaved trained model pipeline to: {model_path}")
    print("Project training complete!")

if __name__ == '__main__':
    main()
