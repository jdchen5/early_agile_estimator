# pipeline.py
"""
Data preprocessing pipeline for machine learning using scikit-learn transformers: 
designed specifically for processing project effort estimation data with a target column like 'project_prf_normalised_work_effort':
    - DataFrame validation
    - Column standardization, missing value imputation, categorical encoding (including multi-value semicolon-separated columns)
    - Data cleaning in a systematic way
    - The main entry point is preprocess_dataframe() which applies all transformations and returns a clean, ML-ready dataset
    - The pipeline uses one-hot encoding for both regular categorical variables and multi-label columns (semicolon-separated values).
"""

import os
import re
import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from constants import FileConstants, PipelineConstants, DataConstants
from config_loader import ConfigLoader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === 1. DataFrameValidator: Validate input DataFrame and check target column ===
class DataFrameValidator(BaseEstimator, TransformerMixin):
    """
    Validate input DataFrame and perform initial checks:
    - Ensures input is a DataFrame
    - Validates target column exists with smart matching
    - Stores original shape and target column info
    """
    
    def __init__(self, target_col: str = 'project_prf_normalised_work_effort'):
        self.target_col = target_col
        self.original_shape = None
        self.original_target_col = None
        
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Convert column name to standardized format"""
        return str(col_name).strip().lower().replace(' ', '_')
    
    def _find_target_column(self, df_columns) -> Union[str, Tuple[None, List[str]]]:
        """Smart target column finder - handles various formats"""
        target_standardized = self._standardize_column_name(self.target_col)
        
        # Try exact match first
        if self.target_col in df_columns:
            return self.target_col
            
        # Try standardized versions of all columns
        for col in df_columns:
            col_standardized = self._standardize_column_name(col)
            if col_standardized == target_standardized:
                return col
                
        # If not found, look for partial matches
        similar_cols = []
        target_words = set(target_standardized.split('_'))
        for col in df_columns:
            col_words = set(self._standardize_column_name(col).split('_'))
            if len(target_words.intersection(col_words)) >= 2:
                similar_cols.append(col)
                
        return None, similar_cols
    
    def transform(self, X):
        """Validate DataFrame and find target column"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        df = X.copy()
        self.original_shape = df.shape
        logging.info(f"Processing DataFrame with shape: {df.shape}")

        # Standardize ALL object/categorical columns: lowercase and strip
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

        # Smart target column finding (only if target column is provided)
        if self.target_col:
            result = self._find_target_column(df.columns)
            
            if isinstance(result, tuple):  # Not found
                actual_col, similar_cols = result
                error_msg = f"Target column '{self.target_col}' not found in DataFrame."
                if similar_cols:
                    error_msg += f" Similar columns found: {similar_cols}"
                else:
                    error_msg += f" Available columns: {list(df.columns)}"
                logging.warning(error_msg)
                # For prediction pipeline, we might not have target column
                self.original_target_col = None
            else:
                actual_col = result
                # Store the original column name we found
                self.original_target_col = actual_col
                
                if actual_col != self.target_col:
                    logging.info(f"Target column found: '{actual_col}' -> will be standardized to '{self.target_col}'")
            
        return df

# === 2. ColumnNameStandardizer: Clean and standardize column names ===
class ColumnNameStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardize column names for consistency:
    - Strips spaces, lowercases, replaces & with _&_, removes special chars
    - Maintains mapping for reference
    """
    
    def __init__(self, target_col: Optional[str] = None, original_target_col: Optional[str] = None):
        self.column_mapping = {}
        self.target_col = target_col
        self.original_target_col = original_target_col
        
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def _standardize_columns(self, columns) -> List[str]:
        """Standardize column names"""
        return [str(col).strip().lower().replace(' ', '_') for col in columns]
    
    def _clean_column_names(self, columns) -> List[str]:
        """Clean column names for compatibility"""
        cleaned_cols = []
        for col in columns:
            # Replace ampersands with _&_ to match expected transformations
            col_clean = str(col).replace(' & ', '_&_')
            # Remove special characters except underscores and ampersands
            col_clean = re.sub(r'[^\w\s&]', '', col_clean)
            # Replace spaces with underscores
            col_clean = col_clean.replace(' ', '_')
            cleaned_cols.append(col_clean)
        return cleaned_cols
    
    def transform(self, X):
        """Apply column name standardization"""
        df = X.copy()
        
        # Store original column names
        original_columns = df.columns.tolist()
        
        # Apply standardization
        standardized_cols = self._standardize_columns(original_columns)
        cleaned_cols = self._clean_column_names(standardized_cols)

        # Special handling for target column
        if self.original_target_col and self.target_col:
            try:
                target_index = original_columns.index(self.original_target_col)
                cleaned_cols[target_index] = self.target_col
                logging.info(f"Target column '{self.original_target_col}' -> '{self.target_col}'")
            except ValueError:
                pass  # Original target col not found, proceed normally
        
        # Create mapping
        self.column_mapping = dict(zip(original_columns, cleaned_cols))
        
        # Apply new column names
        df.columns = cleaned_cols
        
        # Report changes
        changed_cols = sum(1 for orig, new in self.column_mapping.items() if orig != new)
        logging.info(f"Standardized {changed_cols} column names")
        
        return df

# === 3. CategoricalValueStandardizer: Apply standardization mapping to categorical values ===
class CategoricalValueStandardizer(BaseEstimator, TransformerMixin):
    """Apply standardization mapping to categorical column values"""
    
    def __init__(self, mapping: Optional[Dict[str, str]] = None, columns: Optional[List[str]] = None):
        self.mapping = mapping or {}
        self.columns = columns

    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        if self.columns is None:
            possible_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            single_value_cols = []
            for col in possible_cols:
                if not X[col].dropna().astype(str).str.contains(';').any():
                    single_value_cols.append(col)
            self.columns = single_value_cols
            logging.info(f"Single-value categorical columns selected for mapping: {self.columns}")
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.strip()
                    .str.lower()
                    .map(lambda x: self.mapping.get(x, x))
                )
        return df

# === 4. CategoricalValueCleaner: Clean categorical values ===
class CategoricalValueCleaner(BaseEstimator, TransformerMixin):
    """
    Clean categorical column values:
    - Replace '-' with '_'
    - Lowercase and strip whitespace
    """
    
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
        
    def transform(self, X):
        df = X.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace('-', '_')
                .str.lower()
                .str.strip()
            )
        return df

# === 5. MissingValueAnalyzer: Analyze and handle missing values ===
class MissingValueAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analyze and handle missing values:
    - Reports missing value statistics
    - Drops high-missing columns (except protected ones)
    - Fills remaining missing values appropriately
    """
    
    def __init__(self, high_missing_threshold: float = 0.7, cols_to_keep: Optional[List[str]] = None):
        self.high_missing_threshold = high_missing_threshold
        self.cols_to_keep = cols_to_keep or []
        self.high_missing_cols = []
        self.missing_stats = {}
        self.fill_values = {}
        
    def fit(self, X, y=None):
        # Pre-calculate fill values for numeric columns
        num_cols = X.select_dtypes(include=['number']).columns
        self.fill_values = {col: X[col].median() for col in num_cols}
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def transform(self, X):
        """Analyze and handle missing values"""
        df = X.copy()
        
        # Calculate missing percentages
        missing_pct = df.isnull().mean()
        self.missing_stats = missing_pct.sort_values(ascending=False)
        
        logging.info(f"Missing value analysis: >50%: {sum(missing_pct > 0.5)}, >70%: {sum(missing_pct > self.high_missing_threshold)}")
        
        # Identify high missing columns
        self.high_missing_cols = missing_pct[missing_pct > self.high_missing_threshold].index.tolist()
        
        # Filter out protected columns
        cols_to_drop = [col for col in self.high_missing_cols if col not in self.cols_to_keep]
        
        if cols_to_drop:
            logging.info(f"Dropping {len(cols_to_drop)} columns with >{self.high_missing_threshold*100}% missing values")
            df = df.drop(columns=cols_to_drop)
        
        # Fill missing values efficiently
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('missing')
        
        # Numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            if col in self.fill_values and df[col].isnull().any():
                df[col] = df[col].fillna(self.fill_values[col])
        
        logging.info(f"Data shape after missing value handling: {df.shape}")
        return df

# === 6. SemicolonProcessor: Process multi-value columns ===
class SemicolonProcessor(BaseEstimator, TransformerMixin):
    """
    Process semicolon-separated values in columns:
    - Identifies columns with semicolons
    - Cleans, deduplicates, and sorts values
    - Applies standardization mapping where specified
    """
    
    def __init__(self, standardization_mapping: Optional[Dict[str, str]] = None):
        self.semicolon_cols = []
        self.standardization_mapping = standardization_mapping or {}
        
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def _clean_and_sort_semicolon(self, val, apply_standardization: bool = False, 
                                 mapping: Optional[Dict[str, str]] = None) -> str:
        """Clean, deduplicate, sort, and standardize semicolon-separated values"""
        if pd.isnull(val) or val == '':
            return val
        
        parts = [x.strip().lower() for x in str(val).split(';') if x.strip()]
        
        if apply_standardization and mapping is not None:
            parts = [mapping.get(part, part) for part in parts]
        
        unique_cleaned = sorted(set(parts))
        return '; '.join(unique_cleaned)
    
    def transform(self, X):
        """Process semicolon-separated columns"""
        df = X.copy()
        # Identify columns with semicolons
        self.semicolon_cols = [
            col for col in df.columns
            if df[col].dropna().astype(str).str.contains(';').any()
        ]
        logging.info(f"Found {len(self.semicolon_cols)} columns with semicolons: {self.semicolon_cols}")
        
        # Process each semicolon column
        for col in self.semicolon_cols:
            apply_mapping = col in self.semicolon_cols
            mapping = self.standardization_mapping if apply_mapping else None
            df[col] = df[col].apply(
                lambda x: self._clean_and_sort_semicolon(
                    x, apply_standardization=apply_mapping, mapping=mapping
                )
            )
        return df

# === 7. MultiValueEncoder: Encode semicolon columns using MultiLabelBinarizer ===
class MultiValueEncoder(BaseEstimator, TransformerMixin):
    """
    Handle multi-value columns using MultiLabelBinarizer:
    - Only processes columns with manageable cardinality
    - Creates binary columns for each unique value
    """
    
    def __init__(self, max_cardinality: int = 10):
        self.max_cardinality = max_cardinality
        self.multi_value_cols = []
        self.mlb_transformers = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Encode multi-value columns"""
        df = X.copy()
        
        # Identify semicolon columns
        semicolon_cols = [
            col for col in df.columns
            if df[col].dropna().astype(str).str.contains(';').any()
        ]
        
        # Filter for low cardinality multi-value columns
        self.multi_value_cols = []
        for col in semicolon_cols:
            # Get unique values across all entries
            all_values = set()
            for val in df[col].dropna().astype(str):
                values = [v.strip() for v in val.split(';') if v.strip()]
                all_values.update(values)
            
            if len(all_values) <= self.max_cardinality:
                self.multi_value_cols.append(col)
        
        logging.info(f"Encoding {len(self.multi_value_cols)} multi-value columns: {self.multi_value_cols}")
        
        # Process each multi-value column
        for col in self.multi_value_cols:
            # Prepare data for MultiLabelBinarizer
            values_list = []
            for idx in df.index:
                val = df.loc[idx, col]
                if pd.notna(val) and str(val).strip():
                    values_list.append([item.strip() for item in str(val).split(';') if item.strip()])
                else:
                    values_list.append([])
            
            if not any(values_list):  # Skip if no valid values
                continue
                
            # Fit and transform
            mlb = MultiLabelBinarizer()
            onehot_array = mlb.fit_transform(values_list)
            
            # Create DataFrame with proper column names
            onehot_df = pd.DataFrame(
                onehot_array,
                columns=[f"{col}__{cat}" for cat in mlb.classes_],
                index=df.index
            )
            
            # Store transformer
            self.mlb_transformers[col] = mlb
            
            # Check for column conflicts and resolve
            overlap = df.columns.intersection(onehot_df.columns)
            if not overlap.empty:
                logging.info(f"Resolving column conflicts for {col}: {list(overlap)}")
                onehot_df = onehot_df.drop(columns=overlap)
            
            # Join with main dataframe
            df = pd.concat([df, onehot_df], axis=1)
            
            logging.info(f"Encoded {col} into {len(mlb.classes_)} binary columns")
        
        # Remove original multi-value columns
        df = df.drop(columns=self.multi_value_cols)
        
        return df

# === 8. CategoricalEncoder: One-hot encode regular categorical columns ===
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Handle single-value categorical columns:
    - Excludes semicolon columns
    - Only encodes low-cardinality columns
    - Optionally drops first category to avoid multicollinearity
    """
    
    def __init__(self, max_cardinality: int = 10, drop_first: bool = False):
        self.max_cardinality = max_cardinality
        self.drop_first = drop_first
        self.categorical_cols = []
        
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def transform(self, X):
        """Encode categorical columns"""
        df = X.copy()
        
        # Identify categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude semicolon(multi-value) columns 
        semicolon_cols = [
            col for col in df.columns
            if df[col].dropna().astype(str).str.contains(';').any()
        ]
        
        # Filter for low cardinality single-value categorical columns
        self.categorical_cols = [
            col for col in cat_cols 
            if col not in semicolon_cols and df[col].nunique() <= self.max_cardinality
        ]
        
        logging.info(f"One-hot encoding {len(self.categorical_cols)} categorical columns: {self.categorical_cols}")

        # Apply one-hot encoding
        if self.categorical_cols:
            df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=self.drop_first)
        
        return df

# === 9. ColumnNameFixer: Final column name cleanup ===
class ColumnNameFixer(BaseEstimator, TransformerMixin):
    """
    Fix column names for compatibility:
    - Removes illegal characters
    - Handles duplicates
    - Ensures clean, consistent naming
    """
    
    def __init__(self):
        self.column_transformations = {}
        
    def fit(self, X, y=None):
        logging.info(f"{self.__class__.__name__}.fit() CALLED")
        return self
    
    def transform(self, X):
        """Fix problematic column names"""
        df = X.copy()
        original_cols = df.columns.tolist()
        fixed_columns = []
        seen_columns = set()
        
        for col in original_cols:
            # Clean column name
            fixed_col = str(col).replace(' ', '_').replace('&', 'and')
            fixed_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in fixed_col)
            fixed_col = re.sub('_+', '_', fixed_col).strip('_')
            
            # Handle duplicates
            base_col = fixed_col
            suffix = 1
            while fixed_col in seen_columns:
                fixed_col = f"{base_col}_{suffix}"
                suffix += 1
            
            seen_columns.add(fixed_col)
            fixed_columns.append(fixed_col)
        
        # Store transformations
        self.column_transformations = dict(zip(original_cols, fixed_columns))
        df.columns = fixed_columns
        
        n_changed = sum(1 for old, new in self.column_transformations.items() if old != new)
        logging.info(f"Fixed {n_changed} column names for compatibility")
        
        return df

# === 10. DataValidator: Final validation and summary ===
class DataValidator(BaseEstimator, TransformerMixin):
    """
    Validate final dataset:
    - Check shape, missing values, data types
    - Provide target variable summary
    - Report any issues
    """
    
    def __init__(self, target_col: str):
        self.target_col = target_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Validate the processed dataset"""
        df = X.copy()
        
        logging.info(f"=== Final Data Validation ===")
        logging.info(f"Final shape: {df.shape}")
        logging.info(f"Target column: {self.target_col}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        logging.info(f"Total missing values: {missing_count}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(df[numeric_cols].values).sum()
            logging.info(f"Total infinite values: {inf_count}")
        
        # Data types summary
        logging.info(f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        logging.info(f"Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
        
        # Target variable summary
        if self.target_col and self.target_col in df.columns:
            target_stats = df[self.target_col].describe()
            logging.info(f"Target variable '{self.target_col}' statistics: Count: {target_stats['count']}, Mean: {target_stats['mean']:.2f}")
        
        return df

# === Pipeline creation function ===
def create_preprocessing_pipeline(
    target_col: str = None,  # Made optional for prediction pipeline
    high_missing_threshold: float = 0.7,
    cols_to_keep: Optional[List[str]] = None,
    max_categorical_cardinality: int = 10,
    standardization_mapping: Optional[Dict[str, str]] = None
) -> Pipeline:
    """
    Create complete preprocessing pipeline for DataFrame input
    
    Parameters:
    -----------
    target_col : str, optional
        Name of target column (optional for prediction pipeline)
    high_missing_threshold : float
        Threshold for dropping columns with high missing values
    cols_to_keep : list
        Columns to keep even if they have high missing values
    max_categorical_cardinality : int
        Maximum number of unique values for categorical encoding
    standardization_mapping : dict
        Custom mapping for standardizing values
    
    Returns:
    --------
    Pipeline
        Complete preprocessing pipeline
    """
    
    if cols_to_keep is None:
        cols_to_keep = [
            'project_prf_case_tool_used', 
            'process_pmf_prototyping_used',
            'tech_tf_client_roles', 
            'tech_tf_type_of_server', 
            'tech_tf_clientserver_description'
        ]
    
    pipeline = Pipeline([
        ('validator', DataFrameValidator(target_col)),
        ('column_standardizer', ColumnNameStandardizer()),
        ('missing_handler', MissingValueAnalyzer(
            high_missing_threshold=high_missing_threshold,
            cols_to_keep=cols_to_keep
        )),
        ('cat_value_cleaner', CategoricalValueCleaner()),
        ('semicolon_processor', SemicolonProcessor(standardization_mapping=standardization_mapping)),
        ('cat_value_standardizer', CategoricalValueStandardizer(
            mapping=standardization_mapping,
            columns=None
        )),
        ('multi_value_encoder', MultiValueEncoder(max_cardinality=max_categorical_cardinality)),
        ('categorical_encoder', CategoricalEncoder(max_cardinality=max_categorical_cardinality)),
        ('column_fixer', ColumnNameFixer()),
        ('final_validator', DataValidator(target_col)),
        ('schema_aligner', SchemaAligner()) 
    ])
    
    return pipeline

# === Feature-to-DataFrame converter for prediction ===
def convert_feature_dict_to_dataframe(feature_dict: Dict, feature_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convert feature dictionary from UI to DataFrame suitable for preprocessing pipeline.
    
    Parameters:
    -----------
    feature_dict : dict
        Feature dictionary from UI (with one-hot encoded features)
    feature_config : dict, optional
        Feature configuration to understand structure
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original categorical structure suitable for preprocessing
    """
    if feature_config is None:
        feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        feature_config = ConfigLoader.load_yaml_config(feature_config_path)
        if feature_config is None:
            feature_config = {}
    
    # Create DataFrame from feature dict (exclude metadata)
    exclude_keys = {'selected_model', 'submit'}
    clean_features = {k: v for k, v in feature_dict.items() if k not in exclude_keys}
    
    # Convert one-hot features back to categorical
    df_data = {}
    processed_features = set()
    
    # Handle numeric features directly
    for feature in feature_config.get("numeric_features", []):
        if feature in clean_features:
            df_data[feature] = clean_features[feature]
            processed_features.add(feature)
    
    # Handle categorical features directly
    for feature in feature_config.get("categorical_features", {}):
        if feature in clean_features:
            df_data[feature] = clean_features[feature]
            processed_features.add(feature)
    
    # Convert one-hot features back to categorical
    for group_name, group_config in feature_config.get("one_hot_features", {}).items():
        input_key = group_config.get("input_key")
        mapping = group_config.get("mapping", {})
        
        # Find which one-hot features are active
        active_values = []
        for label, feature_key in mapping.items():
            if feature_key in clean_features and clean_features[feature_key] == 1:
                active_values.append(label)
                processed_features.add(feature_key)
        
        # Set the categorical value
        if active_values:
            if len(active_values) == 1:
                df_data[input_key] = active_values[0]
            else:
                # Multiple values - join with semicolon for multi-value fields
                df_data[input_key] = ';'.join(active_values)
        else:
            df_data[input_key] = 'missing'  # Default value
    
    # Handle special cases (like team size group)
    for group_name, group_config in feature_config.get("special_cases", {}).items():
        input_key = group_config.get("input_key")
        output_keys = group_config.get("output_keys", {})
        
        # Find which special case feature is active
        for label, feature_key in output_keys.items():
            if feature_key in clean_features and clean_features[feature_key] == 1:
                df_data[input_key] = label
                processed_features.add(feature_key)
                break
        else:
            # No active feature found, use default
            if input_key not in df_data:
                df_data[input_key] = 'Missing'
    
    # Handle binary features
    for group_name, group_config in feature_config.get("binary_features", {}).items():
        input_key = group_config.get("input_key")
        mapping = group_config.get("mapping", {})
        
        # Find which binary feature is active
        for label, feature_key in mapping.items():
            if feature_key in clean_features and clean_features[feature_key] == 1:
                df_data[input_key] = label
                processed_features.add(feature_key)
                break
        else:
            # No active feature found, use default
            if input_key not in df_data:
                df_data[input_key] = list(mapping.keys())[0] if mapping else 'No'
    
    # Log any unprocessed features
    unprocessed = set(clean_features.keys()) - processed_features
    if unprocessed:
        logging.warning(f"Unprocessed features: {unprocessed}")

        for feature_name, value in clean_features.items():
            if feature_name not in processed_features:
                df_data[feature_name] = value
                processed_features.add(feature_name)
                logging.info(f"DEBUG: Added unprocessed feature directly: {feature_name} = {value}")
            
    
    # Create DataFrame with single row
    df = pd.DataFrame([df_data])
    
    logging.info(f"Converted feature dict to DataFrame with shape {df.shape} and columns: {list(df.columns)}")
    return df

# === Save and load pipeline ===
def save_preprocessing_pipeline(pipeline: Pipeline, filepath: str = FileConstants.PIPELINE_FILE):
    """Save preprocessing pipeline to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    logging.info(f"Preprocessing pipeline saved to {filepath}")

def load_preprocessing_pipeline(filepath: str = FileConstants.PIPELINE_FILE) -> Optional[Pipeline]:
    """Load preprocessing pipeline from disk"""
    if not os.path.exists(filepath):
        logging.warning(f"Preprocessing pipeline not found at {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info(f"Preprocessing pipeline loaded from {filepath}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading preprocessing pipeline: {e}")
        return None

# === Simplified preprocessing function ===
def preprocess_dataframe(
    df: pd.DataFrame,
    target_col: str = 'project_prf_normalised_work_effort',
    **pipeline_kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess a DataFrame using the complete pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to preprocess
    target_col : str
        Name of target column
    **pipeline_kwargs : dict
        Additional arguments for pipeline creation
    
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame ready for modeling
    dict
        Processing metadata and statistics
    """
    
    logging.info("="*60)
    logging.info("DataFrame Preprocessing Pipeline")
    logging.info("="*60)
    logging.info(f"Input shape: {df.shape}")
    logging.info(f"Target column: {target_col}")
    logging.info(f"Timestamp: {datetime.now()}")
    
    # Create and apply pipeline
    pipeline = create_preprocessing_pipeline(target_col=target_col, **pipeline_kwargs)
    
    # Get original target column info from validator
    validator = pipeline.named_steps['validator']
    df_processed = pipeline.fit_transform(df)
    
    # Prepare metadata
    metadata = {
        'original_shape': df.shape,
        'processed_shape': df_processed.shape,
        'processing_timestamp': datetime.now().isoformat(),
        'target_column_standardized': target_col,
        'target_column_original': validator.original_target_col,
        'pipeline_steps': [step[0] for step in pipeline.steps]
    }
    
    logging.info("\n" + "="*60)
    logging.info("Preprocessing completed successfully!")
    logging.info(f"Shape: {df.shape} -> {df_processed.shape}")
    logging.info("="*60)
    
    return df_processed, metadata

# === Prediction preprocessing function ===
def preprocess_for_prediction(
    feature_dict: Dict,
    pipeline: Optional[Pipeline] = None,
    feature_config: Optional[Dict] = None,
    create_dynamic_pipeline: bool = True  # NEW: Enable dynamic pipeline creation
) -> pd.DataFrame:
    """
    Preprocess feature dictionary for prediction.
    Can work with saved pipeline OR create dynamic pipeline (no pkl file required).
    
    Parameters:
    -----------
    feature_dict : dict
        Feature dictionary from UI
    pipeline : Pipeline, optional
        Fitted preprocessing pipeline. If None, creates dynamic pipeline
    feature_config : dict, optional
        Feature configuration for understanding structure
    create_dynamic_pipeline : bool
        If True, creates pipeline dynamically when no saved pipeline exists
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed features ready for model prediction
    """
    
    # Load feature config if not provided
    if feature_config is None:
        feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        feature_config = ConfigLoader.load_yaml_config(feature_config_path)
        if feature_config is None:
            feature_config = {}
    
    # Try to load saved pipeline first (optional)
    if pipeline is None:
        pipeline = load_preprocessing_pipeline()
        
        # If no saved pipeline and dynamic creation is enabled, create one
        if pipeline is None and create_dynamic_pipeline:
            logging.info("No saved pipeline found. Creating dynamic preprocessing pipeline...")
            pipeline = create_preprocessing_pipeline(
                target_col=None,  # No target for prediction
                high_missing_threshold=0.9,  # More lenient for prediction
                max_categorical_cardinality=20  # More lenient for prediction
            )
            logging.info("Dynamic preprocessing pipeline created successfully")
        
        # If still no pipeline, use fallback
        elif pipeline is None:
            logging.warning("No preprocessing pipeline available and dynamic creation disabled.")
            # Fallback: just convert to DataFrame without full preprocessing
            df = convert_feature_dict_to_dataframe(feature_dict, feature_config)
            return df
    
    # Convert feature dict to DataFrame
    df = convert_feature_dict_to_dataframe(feature_dict, feature_config)
    
    # Apply preprocessing
    try:
        # For prediction, we fit_transform on the single row
        # (This is safe because each transformer handles single-row data appropriately)
        df_processed = pipeline.fit_transform(df)
        
        # Remove target column if it exists (since we're doing prediction)
        target_cols = [col for col in df_processed.columns 
                      if any(keyword in col.lower() for keyword in ['target', 'effort', 'work_effort'])]
        if target_cols:
            df_processed = df_processed.drop(columns=target_cols)
            logging.info(f"Removed target columns for prediction: {target_cols}")
        
        logging.info(f"Preprocessed features for prediction: shape {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logging.error(f"Error in preprocessing for prediction: {e}")
        logging.warning("Using fallback preprocessing...")
        # Fallback: return original DataFrame with basic cleaning
        return df

# Add a simpler version that always uses dynamic pipeline:
def preprocess_for_prediction_dynamic(
    feature_dict: Dict,
    feature_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Simple preprocessing function that always creates pipeline dynamically.
    No pkl file required.
    """
    return preprocess_for_prediction(
        feature_dict=feature_dict,
        pipeline=None,
        feature_config=feature_config,
        create_dynamic_pipeline=True
    )

# === Utility functions ===
def get_pipeline_feature_names(pipeline: Pipeline) -> List[str]:
    """Get the expected feature names from a fitted preprocessing pipeline"""
    try:
        # Create a dummy dataframe and transform it to see output columns
        dummy_data = {}
        
        # Load feature config to create realistic dummy data
        feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        feature_config = ConfigLoader.load_yaml_config(feature_config_path)
        if feature_config is None:
            feature_config = {}
        
        # Add numeric features with default values
        for feature in feature_config.get("numeric_features", []):
            dummy_data[feature] = 0
        
        # Add categorical features with default values
        for feature, config in feature_config.get("categorical_features", {}).items():
            options = config.get("options", [])
            dummy_data[feature] = options[0] if options else "default"
        
        # Add one-hot input fields with default values
        for group_name, group_config in feature_config.get("one_hot_features", {}).items():
            input_key = group_config.get("input_key")
            mapping = group_config.get("mapping", {})
            if mapping:
                dummy_data[input_key] = list(mapping.keys())[0]
        
        # Create dummy DataFrame
        dummy_df = pd.DataFrame([dummy_data])
        
        # Transform and get column names
        transformed_df = pipeline.transform(dummy_df)
        return list(transformed_df.columns)
        
    except Exception as e:
        logging.error(f"Error getting pipeline feature names: {e}")
        return []

def validate_pipeline_compatibility(pipeline: Pipeline, feature_dict: Dict) -> Dict[str, Any]:
    """
    Validate if feature dictionary is compatible with preprocessing pipeline
    
    Returns validation report with compatibility status
    """
    try:
        # Attempt to preprocess the features
        df_processed = preprocess_for_prediction(feature_dict, pipeline)
        
        pipeline_features = get_pipeline_feature_names(pipeline)
        processed_features = list(df_processed.columns)
        
        missing_features = set(pipeline_features) - set(processed_features)
        extra_features = set(processed_features) - set(pipeline_features)
        
        return {
            'compatible': len(missing_features) == 0,
            'processed_shape': df_processed.shape,
            'expected_features': len(pipeline_features),
            'actual_features': len(processed_features),
            'missing_features': list(missing_features),
            'extra_features': list(extra_features),
            'validation_passed': len(missing_features) == 0 and len(extra_features) == 0
        }
        
    except Exception as e:
        return {
            'compatible': False,
            'error': str(e),
            'validation_passed': False
        }



# === Main functions for external use ===
def create_dynamic_pipeline_for_prediction(feature_dict: Dict) -> Pipeline:
    """
    Create a dynamic preprocessing pipeline specifically for prediction use.
    This is the main function to use when you don't have a saved pipeline.
    """
    feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
    feature_config = ConfigLoader.load_yaml_config(feature_config_path)
    if feature_config is None:
            feature_config = {}
    
    # Create pipeline optimized for prediction
    pipeline = create_preprocessing_pipeline(
        target_col=None,  # No target column for prediction
        high_missing_threshold=0.95,  # Very lenient for prediction
        max_categorical_cardinality=50,  # More lenient for prediction
        standardization_mapping=None
    )
    
    logging.info("Created dynamic pipeline for prediction")
    return pipeline

def process_features_for_prediction(feature_dict: Dict) -> pd.DataFrame:
    """
    Main entry point for processing features for prediction.
    Creates pipeline dynamically and processes the features.
    """
    try:
        # Create dynamic pipeline
        pipeline = create_dynamic_pipeline_for_prediction(feature_dict)
        
        # Convert features to DataFrame
        feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        feature_config = ConfigLoader.load_yaml_config(feature_config_path)
        if feature_config is None:
            feature_config = {}
        df = convert_feature_dict_to_dataframe(feature_dict, feature_config)
        
        # Process with pipeline
        df_processed = pipeline.fit_transform(df)
        
        # Remove any target-related columns
        target_cols = [col for col in df_processed.columns 
                      if any(keyword in col.lower() for keyword in ['target', 'effort', 'work_effort'])]
        if target_cols:
            df_processed = df_processed.drop(columns=target_cols)
            logging.info(f"Removed target columns: {target_cols}")
        
        logging.info(f"Successfully processed features: shape {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logging.error(f"Error in process_features_for_prediction: {e}")
        # Fallback: basic DataFrame conversion
        feature_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
        feature_config = ConfigLoader.load_yaml_config(feature_config_path)
        if feature_config is None:
            feature_config = {}
        df = convert_feature_dict_to_dataframe(feature_dict, feature_config)
        return df

class SchemaAligner(BaseEstimator, TransformerMixin):
    """
    Align features to match training schema from full_model_features.json
    Takes current pipeline output and expands it to complete feature set.
    """
    
    def __init__(self, schema_file: str = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.DYNAMIC_FULL_MODEL_FEATURES_FILE)):
        self.schema_file = schema_file
        self.expected_features = None
        
    def fit(self, X, y=None):
        # Load expected features from training schema
        try:
            import json
            with open(self.schema_file, 'r') as f:
                self.expected_features = json.load(f)
            logging.info(f"SchemaAligner: Loaded {len(self.expected_features)} expected features")
        except Exception as e:
            logging.error(f"SchemaAligner: Could not load schema file {self.schema_file}: {e}")
            self.expected_features = []
        return self
    
    def transform(self, X):
        """Align current features to training schema"""
        if not self.expected_features:
            logging.warning("SchemaAligner: No expected features loaded, returning original data")
            return X
            
        df = X.copy()
        
        # Add missing columns with 0 values
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training schema and remove extras
        df_aligned = df.reindex(columns=self.expected_features, fill_value=0)
        # Add target column with dummy value for prediction
        if 'project_prf_normalised_work_effort' in self.expected_features:
            df_aligned['project_prf_normalised_work_effort'] = 0
        
        logging.info(f"SchemaAligner: {X.shape} → {df_aligned.shape}")
        return df_aligned


# === Export functions ===
__all__ = [
    # Main pipeline functions
    'create_preprocessing_pipeline',
    'preprocess_dataframe',
    'preprocess_for_prediction',
    'preprocess_for_prediction_dynamic',
    
    # Feature conversion
    'convert_feature_dict_to_dataframe',
    'process_features_for_prediction',
    'create_dynamic_pipeline_for_prediction',
    
    # Pipeline management
    'save_preprocessing_pipeline',
    'load_preprocessing_pipeline',
    
    # Utilities
    'get_pipeline_feature_names',
    'validate_pipeline_compatibility',
    'load_yaml_config',
    
    # Individual transformer classes
    'DataFrameValidator',
    'ColumnNameStandardizer',
    'CategoricalValueStandardizer',
    'CategoricalValueCleaner',
    'MissingValueAnalyzer',
    'SemicolonProcessor',
    'MultiValueEncoder',
    'CategoricalEncoder',
    'ColumnNameFixer',
    'DataValidator'
]