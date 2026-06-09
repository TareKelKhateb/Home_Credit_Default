from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier, CatBoostError
from src.transformers import CategoricalNaNTransformer, DaysEmployedAnomalyTransformer
from src.config import CATBOOST_PARAMS

def create_model_pipeline(categorical_features, custom_params=None):
    """
    Creates and initializes the unified Scikit-Learn Pipeline combining
    custom transformers and the CatBoost Classifier.
    Automatically checks if GPU is supported and falls back to CPU if needed.
    """
    params = CATBOOST_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
        
    # Set categorical feature lists
    params['cat_features'] = categorical_features
    
    # Try initializing CatBoost with GPU task type; fallback to CPU if driver/GPU unavailable.
    try:
        classifier = CatBoostClassifier(**params)
    except CatBoostError as e:
        print(f"CatBoost GPU failed to initialize: {e}")
        print("Falling back to CPU configuration...")
        params['task_type'] = 'CPU'
        params.pop('devices', None)
        classifier = CatBoostClassifier(**params)
        
    model_pipeline = Pipeline(steps=[
        ('cat_nan_handler', CategoricalNaNTransformer()),
        ('days_employed_handler', DaysEmployedAnomalyTransformer()),
        ('classifier', classifier)
    ])
    
    return model_pipeline
