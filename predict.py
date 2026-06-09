import os
import argparse
import joblib
import pandas as pd
from src.config import TARGET

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Predictions using Trained CatBoost Pipeline")
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing the saved model and cached features')
    parser.add_argument('--output', type=str, default='submission.csv', help='Filename for the output predictions CSV')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_path = os.path.join(args.model_dir, 'catboost_pipeline.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
        
    print(f"Loading trained model pipeline from {model_path}...")
    model_pipeline = joblib.load(model_path)
    
    # Check for pre-processed test features cached in train.py
    cached_test_path = os.path.join(args.model_dir, 'temp', 'processed_test_features.joblib')
    if os.path.exists(cached_test_path):
        print("Loading preprocessed test features from cache...")
        test_df = joblib.load(cached_test_path)
    else:
        # Fallback if cache doesn't exist (e.g. inference only run)
        # Note: In production we would load raw test + run preprocess_data(data_path, df_test)
        raise FileNotFoundError(
            f"Preprocessed test features cache not found at {cached_test_path}. "
            "Please run train.py first to cache preprocessed test features, or integrate raw CSV inputs."
        )
        
    print(f"Test data shape: {test_df.shape}")
    
    # 3. Predict probabilities on test set
    print("Generating predictions...")
    features = test_df.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    # Predict default probability (second class)
    probabilities = model_pipeline.predict_proba(features)[:, 1]
    
    # 4. Generate submission dataframe
    submission = pd.DataFrame({
        'SK_ID_CURR': test_df['SK_ID_CURR'],
        'TARGET': probabilities
    })
    
    submission.to_csv(args.output, index=False)
    print(f"Submission file created successfully at: {args.output}")
    print(submission.head())

if __name__ == '__main__':
    main()
