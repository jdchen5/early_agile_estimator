# constants.py
"""
Central constants file for ML Project Effort Estimator
All magic numbers, file paths, and configuration values in one place
"""

class FileConstants:
    # Folder paths
    CONFIG_FOLDER = 'config'
    MODELS_FOLDER = 'models'
    DATA_FOLDER = 'data'
    
    # Configuration files
    UI_INFO_FILE = 'ui_info.yaml'
    FEATURE_MAPPING_FILE = 'feature_mapping.yaml'
    MODEL_DISPLAY_NAME_FILE = 'model_display_names.json'
    SHAP_ANALYSIS_FILE = 'shap_analysis.md'
    DYNAMIC_FULL_MODEL_FEATURES_FILE = 'full_model_features.json'
    
    # Pipeline files
    PIPELINE_FILE = 'preprocessing_pipeline.pkl'
    PIPELINE_MODEL_FILE = 'synthetic_isbsg2016r1_1_finance_sdv_generated_pipeline.pkl'
    
    # Data files
    ISBSG_PREPROCESSED_FILE = 'synthetic_isbsg2016r1_1_finance_sdv_generated_fixed_columns_data.csv'


class PipelineConstants:
    # Pipeline parameters
    HIGH_MISSING_THRESHOLD = 0.7
    MAX_CATEGORICAL_CARDINALITY = 10
    DEFAULT_SAMPLE_SIZE = 100
    KERNEL_EXPLAINER_SAMPLE_SIZE = 50
    
    # Processing limits
    MAX_FEATURES_SHOWN = 15
    PRECISION_DECIMALS = 3
    TOP_N_FEATURES = 10
    MAX_ANALYSIS_POINTS = 20

class UIConstants:
    # UI markers and symbols
    REQUIRED_FIELD_MARKER = "⭐"
    
    # Default values
    DEFAULT_TEAM_SIZE = 5
    DEFAULT_FUNCTIONAL_SIZE = 100
    HOURS_PER_DAY = 8
    DAYS_PER_WEEK = 5
    WEEKS_PER_MONTH = 4.33
    
    # Prediction thresholds
    LOW_PREDICTION_WARNING = 10
    HIGH_PREDICTION_WARNING = 192000
    HIGH_VARIANCE_THRESHOLD = 0.3  # 30%
    
    # SHAP analysis
    SHAP_CACHE_SIZE_WARNING = 50
    WHAT_IF_MIN_POINTS = 5
    WHAT_IF_MAX_POINTS = 20
    WHAT_IF_DEFAULT_POINTS = 10

class ModelConstants:
    # Model file extensions
    MODEL_EXTENSION = '.pkl'
    JOBLIB_EXTENSION = '.joblib'
    
    # Model types for SHAP compatibility
    TREE_MODEL_KEYWORDS = [
        'forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient',
        'randomforest', 'extratrees', 'decisiontree'
    ]
    
    LINEAR_MODEL_KEYWORDS = [
        'linear', 'lasso', 'ridge', 'elastic', 'bayesianridge'
    ]
    
    # Model display name mappings
    MODEL_TYPE_MAPPINGS = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost', 
        'lgb': 'LightGBM',
        'lr': 'Linear Regression',
        'svm': 'Support Vector Machine',
        'dt': 'Decision Tree',
        'nb': 'Naive Bayes',
        'knn': 'K-Nearest Neighbors',
        'ada': 'AdaBoost',
        'gb': 'Gradient Boosting',
        'et': 'Extra Trees'
    }

class DataConstants:
    # Column names and keywords
    TARGET_COLUMN = 'project_prf_normalised_work_effort'
    
    # Target-related keywords for removal
    TARGET_KEYWORDS = ['target', 'effort', 'label', 'prediction', 'actual', 'ground_truth']
    
    # UI exclusion keys
    UI_EXCLUSION_KEYS = {
        'selected_model', 'selected_models', 'submit', 'clear_results', 
        'show_history', 'save_config', 'config_name', 'comparison_mode'
    }
    
    # ISBSG ignore columns
    ISBSG_IGNORE_COLUMNS = [
        'isbsg_project_id', 'external_eef_data_quality_rating', 
        'external_eef_data_quality_rating_b', 'project_prf_normalised_work_effort_level_1',
        'project_prf_normalised_level_1_pdr_ufp', 'project_prf_normalised_pdr_ufp',
        'project_prf_project_elapsed_time', 'people_prf_ba_team_experience_less_than_1_yr',
        'people_prf_ba_team_experience_1_to_3_yr', 'people_prf_ba_team_experience_great_than_3_yr',
        'people_prf_it_experience_less_than_1_yr', 'people_prf_it_experience_1_to_3_yr',
        'people_prf_it_experience_great_than_3_yr', 'people_prf_it_experience_less_than_3_yr',
        'people_prf_it_experience_3_to_9_yr', 'people_prf_it_experience_great_than_9_yr',
        'people_prf_project_manage_experience', 'project_prf_total_project_cost',
        'project_prf_cost_currency', 'project_prf_currency_multiple', 'project_prf_speed_of_delivery',
        'people_prf_project_manage_changes', 'project_prf_defect_density', 'project_prf_manpower_delivery_rate'
    ]
    
    # Mixed type columns for preprocessing
    MIXED_TYPE_COLUMNS = [
        'external_eef_industry_sector',
        'tech_tf_client_roles', 
        'tech_tf_clientserver_description',
        'tech_tf_development_platform_hand_held'
    ]

class LoggingConstants:
    # Log file names
    APP_LOG_FILE = "app.log"
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Log levels by component
    DEFAULT_LOG_LEVEL = "INFO"
    SHAP_LOG_LEVEL = "WARNING"

class ValidationConstants:
    # Validation thresholds
    MIN_PREDICTION_VALUE = 0.1
    MAX_PREDICTION_VALUE = 1000000
    
    # Feature validation
    MIN_FEATURE_COUNT = 5
    MAX_FEATURE_COUNT = 200
    
    # Data quality thresholds
    MAX_MISSING_RATIO = 0.95
    MAX_INFINITE_VALUES = 0
    MEMORY_WARNING_MB = 100