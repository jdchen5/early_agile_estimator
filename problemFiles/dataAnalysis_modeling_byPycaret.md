<span style="color: blue;font-weight: bold; font-size: 40px;">ISBSG Data Analysis & Regression </span>


```python
# # ISBSG Data Analysis and Regression Modeling
# 
# This notebook performs data cleaning, preprocessing, and regression modeling on the ISBSG dataset.

# ## Setup and Environment Configuration

# Install required packages (uncomment if needed)
# !pip install -r requirements.txt
```


```python
# Import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
```

<a id = 'Index:'></a>

# Table of Content

In this notebook you will apply xxxxxxx


- [Part 1](#part1)- Data Loading and Initial Exploration
- [Part 2](#part2)- Data Cleaning and Preprocessing
- [Part 3](#part3)- Data Profiling
- [Part 4](#part4)- Module Building with PyCaret
- [Part 5](#part5)- Model Preparation
- [Part 6](#part6)- Baseline Modeling and Evaluation
- [Part 7](#part7)- Advanced Modeling and Hyperparameter Tuning
- [Part 8](#part8)- Model Comparison and Selection
- [Part 9](#part9)- End



```python
# Configure timestamp callback for Jupyter cells
from IPython import get_ipython

def setup_timestamp_callback():
    """Setup a timestamp callback for Jupyter cells without clearing existing callbacks."""
    ip = get_ipython()
    if ip is not None:
        # Define timestamp function
        def print_timestamp(*args, **kwargs):
            """Print timestamp after cell execution."""
            print(f"Cell executed at: {datetime.now()}")
        
        # Check if our callback is already registered
        callbacks = ip.events.callbacks.get('post_run_cell', [])
        for cb in callbacks:
            if hasattr(cb, '__name__') and cb.__name__ == 'print_timestamp':
                # Already registered
                return
                
        # Register new callback if not already present
        ip.events.register('post_run_cell', print_timestamp)
        print("Timestamp printing activated.")
    else:
        print("Not running in IPython/Jupyter environment.")
```


```python
# Setup timestamp callback
setup_timestamp_callback()

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
```

    Timestamp printing activated.
    Cell executed at: 2025-05-16 16:40:39.457465
    

[Back to top](#Index:)

<a id='part1'></a>

# Part 1 -Data Loading and Initial Exploration

This section is dedicated to loading the dataset, performing initial data exploration such as viewing the first few rows, and summarizing the dataset's characteristics, including missing values and basic statistical measures.


```python
# Load the data
print("Loading data...")
df = pd.read_excel("data/ISBSG2016R1.1-Formatted4CSVAgileOnly.xlsx")

```

    Loading data...
    Cell executed at: 2025-05-16 16:40:42.803495
    


```python
# Clean column names function
def clean_column_names(columns):
    cleaned_cols = []
    for col in columns:
        # First replace ampersands with _&_ to match PyCaret's transformation
        col_clean = col.replace(' & ', '_&_')
        # Then remove any remaining special chars
        col_clean = re.sub(r'[^\w\s&]', '', col_clean)
        # Finally replace spaces with underscores
        col_clean = col_clean.replace(' ', '_')
        cleaned_cols.append(col_clean)
    return cleaned_cols

# Clean column names
original_columns = df.columns.tolist()  # Save original column names for reference
df.columns = clean_column_names(df.columns)
```

    Cell executed at: 2025-05-16 16:40:42.822122
    


```python
# Create a mapping from original to cleaned column names
column_mapping = dict(zip(original_columns, df.columns))
print("\nColumn name mapping (original -> cleaned):")
for orig, clean in column_mapping.items():
    if orig != clean:  # Only show columns that changed
        print(f"  '{orig}' -> '{clean}'")

```

    
    Column name mapping (original -> cleaned):
      'ISBSG Project ID' -> 'ISBSG_Project_ID'
      'External_EEF_Data Quality Rating' -> 'External_EEF_Data_Quality_Rating'
      'Project_PRF_Year of Project' -> 'Project_PRF_Year_of_Project'
      'External_EEF_Industry Sector' -> 'External_EEF_Industry_Sector'
      'External_EEF_Organisation Type' -> 'External_EEF_Organisation_Type'
      'Project_PRF_Application Group' -> 'Project_PRF_Application_Group'
      'Project_PRF_Application Type' -> 'Project_PRF_Application_Type'
      'Project_PRF_Development Type' -> 'Project_PRF_Development_Type'
      'Tech_TF_Development Platform' -> 'Tech_TF_Development_Platform'
      'Tech_TF_Language Type' -> 'Tech_TF_Language_Type'
      'Tech_TF_Primary Programming Language' -> 'Tech_TF_Primary_Programming_Language'
      'Project_PRF_Functional Size' -> 'Project_PRF_Functional_Size'
      'Project_PRF_Relative Size' -> 'Project_PRF_Relative_Size'
      'Project_PRF_Normalised Work Effort Level 1' -> 'Project_PRF_Normalised_Work_Effort_Level_1'
      'Project_PRF_Normalised Work Effort' -> 'Project_PRF_Normalised_Work_Effort'
      'Project_PRF_Normalised Level 1 PDR_ufp' -> 'Project_PRF_Normalised_Level_1_PDR_ufp'
      'Project_PRF_Normalised PDR_ufp' -> 'Project_PRF_Normalised_PDR_ufp'
      'Project_PRF_Defect Density' -> 'Project_PRF_Defect_Density'
      'Project_PRF_Speed of Delivery' -> 'Project_PRF_Speed_of_Delivery'
      'Project_PRF_Manpower Delivery Rate' -> 'Project_PRF_Manpower_Delivery_Rate'
      'Project_PRF_Project Elapsed Time' -> 'Project_PRF_Project_Elapsed_Time'
      'Project_PRF_Team Size Group' -> 'Project_PRF_Team_Size_Group'
      'Project_PRF_Max Team Size' -> 'Project_PRF_Max_Team_Size'
      'Project_PRF_CASE Tool Used' -> 'Project_PRF_CASE_Tool_Used'
      'Process_PMF_Development Methodologies' -> 'Process_PMF_Development_Methodologies'
      'Process_PMF_Prototyping Used' -> 'Process_PMF_Prototyping_Used'
    Cell executed at: 2025-05-16 16:40:42.833957
    


```python
# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

```

    Dataset shape: (7058, 51)
    
    First 5 rows:
       ISBSG_Project_ID External_EEF_Data_Quality_Rating  \
    0             10003                                B   
    1             10011                                B   
    2             10012                                B   
    3             10014                                B   
    4             10015                                B   
    
       Project_PRF_Year_of_Project External_EEF_Industry_Sector  \
    0                         2015                Communication   
    1                         1996                 Construction   
    2                         2002           Wholesale & Retail   
    3                         2004                          NaN   
    4                         2000           Wholesale & Retail   
    
      External_EEF_Organisation_Type Project_PRF_Application_Group  \
    0            Telecommunications;          Business Application   
    1                  Construction;          Business Application   
    2                       Billing;          Business Application   
    3                            NaN                           NaN   
    4      Wholesale & Retail Trade;          Business Application   
    
            Project_PRF_Application_Type Project_PRF_Development_Type  \
    0                    Online. eSales;                  Enhancement   
    1  Stock control & order processing;              New Development   
    2                           Billing;                  Enhancement   
    3                                NaN                  Enhancement   
    4     Management Information System;                  Enhancement   
    
      Tech_TF_Development_Platform Tech_TF_Language_Type  ...  \
    0                        Multi                   NaN  ...   
    1                        Multi                   4GL  ...   
    2                          NaN                   3GL  ...   
    3                          NaN                   NaN  ...   
    4                           MF                   3GL  ...   
    
      People_PRF_IT_experience_great_than_3_yr  \
    0                                      NaN   
    1                                      NaN   
    2                                      NaN   
    3                                      NaN   
    4                                      NaN   
    
       People_PRF_IT_experience_less_than_3_yr People_PRF_IT_experience_3_to_9_yr  \
    0                                      NaN                                NaN   
    1                                      NaN                                NaN   
    2                                      NaN                                NaN   
    3                                      NaN                                NaN   
    4                                      NaN                                NaN   
    
       People_PRF_IT_experience_great_than_9_yr  \
    0                                       NaN   
    1                                       NaN   
    2                                       NaN   
    3                                       NaN   
    4                                       NaN   
    
       People_PRF_Project_manage_experience  People_PRF_Project_manage_changes  \
    0                                   NaN                                NaN   
    1                                   NaN                                NaN   
    2                                   NaN                                NaN   
    3                                   NaN                                NaN   
    4                                   NaN                                NaN   
    
       People_PRF_Personnel_changes  Project_PRF_Total_project_cost  \
    0                           NaN                             NaN   
    1                           NaN                             NaN   
    2                           NaN                             NaN   
    3                           NaN                             NaN   
    4                           NaN                             NaN   
    
       Project_PRF_Cost_currency  Project_PRF_Currency_multiple  
    0                        NaN                            NaN  
    1                        NaN                            NaN  
    2                        NaN                            NaN  
    3                        NaN                            NaN  
    4                        NaN                            NaN  
    
    [5 rows x 51 columns]
    Cell executed at: 2025-05-16 16:40:42.850325
    


```python
# Create a function to get comprehensive data summary
def get_data_summary(df, n_unique_samples=5):
    """
    Generate a comprehensive summary of the dataframe.
    
    Args:
        df: Pandas DataFrame
        n_unique_samples: Number of unique values to show as sample
        
    Returns:
        DataFrame with summary information
    """
    # Summary dataframe with basic info
    summary = pd.DataFrame({
        'Feature': df.columns,
        'data_type': df.dtypes.values,
        'Null_number': df.isnull().sum().values,
        'Null_pct': (df.isnull().mean() * 100).values,
        'Unique_counts': df.nunique().values,
        'unique_samples': [list(df[col].dropna().unique()[:n_unique_samples]) for col in df.columns]
    })
    
    return summary

# Generate and display data summary
summary_df = get_data_summary(df)
print("\nData Summary (first 10 columns):")
print(summary_df.head(10))

```

    
    Data Summary (first 10 columns):
                                Feature data_type  Null_number   Null_pct  \
    0                  ISBSG_Project_ID     int64            0   0.000000   
    1  External_EEF_Data_Quality_Rating    object            0   0.000000   
    2       Project_PRF_Year_of_Project     int64            0   0.000000   
    3      External_EEF_Industry_Sector    object         1222  17.313687   
    4    External_EEF_Organisation_Type    object         1205  17.072825   
    5     Project_PRF_Application_Group    object         2096  29.696798   
    6      Project_PRF_Application_Type    object         1467  20.784925   
    7      Project_PRF_Development_Type    object            0   0.000000   
    8      Tech_TF_Development_Platform    object         1860  26.353075   
    9             Tech_TF_Language_Type    object         1292  18.305469   
    
       Unique_counts                                     unique_samples  
    0           7055                [10003, 10011, 10012, 10014, 10015]  
    1              2                                             [B, A]  
    2             27                     [2015, 1996, 2002, 2004, 2000]  
    3             17  [Communication, Construction, Wholesale & Reta...  
    4            179  [Telecommunications;, Construction;, Billing;,...  
    5              6  [Business Application, Mathematically-Intensiv...  
    6            539  [Online. eSales;, Stock control & order proces...  
    7              3     [Enhancement, New Development, Re-development]  
    8              6                   [Multi, MF, PC, MR, Proprietary]  
    9              6                          [4GL, 3GL, ApG, 2GL, APG]  
    Cell executed at: 2025-05-16 16:40:42.962081
    


```python
# Display descriptive statistics for numeric columns
desc_stats = df.describe().T
print("\nDescriptive Statistics (first 5 rows):")
print(desc_stats.head())
```

    
    Descriptive Statistics (first 5 rows):
                                                 count          mean  \
    ISBSG_Project_ID                            7058.0  21335.873052   
    Project_PRF_Year_of_Project                 7058.0   2004.763814   
    Project_PRF_Functional_Size                 6150.0    316.933496   
    Project_PRF_Normalised_Work_Effort_Level_1  6288.0   3888.236164   
    Project_PRF_Normalised_Work_Effort          7058.0   4314.558373   
    
                                                         std      min      25%  \
    ISBSG_Project_ID                             6545.814378  10003.0  15655.5   
    Project_PRF_Year_of_Project                     5.892202   1989.0   2000.0   
    Project_PRF_Functional_Size                   640.302478      2.0     64.0   
    Project_PRF_Normalised_Work_Effort_Level_1   8674.167063      4.0    619.0   
    Project_PRF_Normalised_Work_Effort          10076.657930      4.0    687.0   
    
                                                    50%       75%       max  
    ISBSG_Project_ID                            21367.5  26920.50   32767.0  
    Project_PRF_Year_of_Project                  2004.0   2010.00    2015.0  
    Project_PRF_Functional_Size                   137.0    323.00   16148.0  
    Project_PRF_Normalised_Work_Effort_Level_1   1550.5   3847.25  230514.0  
    Project_PRF_Normalised_Work_Effort           1666.0   4106.75  266946.0  
    Cell executed at: 2025-05-16 16:40:43.053874
    


```python
# Identify target column
target_col = 'Project_PRF_Normalised_Work_Effort'
print(f"\nTarget variable: '{target_col}'")
```

    
    Target variable: 'Project_PRF_Normalised_Work_Effort'
    Cell executed at: 2025-05-16 16:40:43.062904
    

[Back to top](#Index:)

<a id='part2'></a>

# Part 2 - Data Cleaning and Preprocessing

Here, data cleaning tasks like handling missing values and providing a detailed summary of each feature, including its type, number of unique values, and a preview of unique values, are performed.


```python
# Analyse missing values
print("\nAnalysing missing values...")
missing_pct = df.isnull().mean() * 100
missing_sorted = missing_pct.sort_values(ascending=False)
print("Top 10 columns with highest missing percentages:")
print(missing_sorted.head(10))
```

    
    Analysing missing values...
    Top 10 columns with highest missing percentages:
    People_PRF_IT_experience_less_than_1_yr         99.192406
    People_PRF_IT_experience_1_to_3_yr              99.036554
    People_PRF_IT_experience_great_than_3_yr        98.838198
    Tech_TF_Server_Roles                            96.217059
    People_PRF_IT_experience_less_than_3_yr         96.132049
    People_PRF_BA_team_experience_less_than_1_yr    96.117880
    Tech_TF_Client_Roles                            96.047039
    People_PRF_BA_team_experience_1_to_3_yr         95.834514
    People_PRF_IT_experience_great_than_9_yr        95.792009
    People_PRF_IT_experience_3_to_9_yr              95.168603
    dtype: float64
    Cell executed at: 2025-05-16 16:40:43.085790
    


```python
# Identify columns with high missing values (>70%)
high_missing_cols = missing_pct[missing_pct > 70].index.tolist()
print(f"\nColumns with >70% missing values ({len(high_missing_cols)} columns):")
for col in high_missing_cols[:5]:  # Show first 5
    print(f"  - {col}: {missing_pct[col]:.2f}% missing")
if len(high_missing_cols) > 5:
    print(f"  - ... and {len(high_missing_cols) - 5} more columns")
```

    
    Columns with >70% missing values (23 columns):
      - Project_PRF_Defect_Density: 79.40% missing
      - Project_PRF_Manpower_Delivery_Rate: 71.39% missing
      - Process_PMF_Prototyping_Used: 85.62% missing
      - Tech_TF_Client_Roles: 96.05% missing
      - Tech_TF_Server_Roles: 96.22% missing
      - ... and 18 more columns
    Cell executed at: 2025-05-16 16:40:43.097170
    


```python
# Create a clean dataframe by dropping high-missing columns
df_clean = df.drop(columns=high_missing_cols)
print(f"\nData shape after dropping high-missing columns: {df_clean.shape}")
```

    
    Data shape after dropping high-missing columns: (7058, 28)
    Cell executed at: 2025-05-16 16:40:43.102991
    


```python
# Handle remaining missing values
print("\nHandling remaining missing values...")
```

    
    Handling remaining missing values...
    Cell executed at: 2025-05-16 16:40:43.117416
    


```python
# Fill missing values in categorical columns with "Missing"
cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df_clean[col].fillna('Missing', inplace=True)
```

    Cell executed at: 2025-05-16 16:40:43.156088
    


```python
# Check remaining missing values
remaining_missing = df_clean.isnull().sum()
remaining_missing_count = sum(remaining_missing > 0)
print(f"\nColumns with remaining missing values: {remaining_missing_count}")
if remaining_missing_count > 0:
    print("Top columns with missing values:")
    print(remaining_missing[remaining_missing > 0].sort_values(ascending=False).head())
```

    
    Columns with remaining missing values: 8
    Top columns with missing values:
    Project_PRF_Max_Team_Size                 4817
    Tech_TF_Tools_Used                        2618
    Project_PRF_Normalised_Level_1_PDR_ufp    1648
    Project_PRF_Speed_of_Delivery             1619
    Project_PRF_Functional_Size                908
    dtype: int64
    Cell executed at: 2025-05-16 16:40:43.185421
    


```python
# Verify target variable
print(f"\nTarget variable '{target_col}' summary:")
print(f"Unique values: {df_clean[target_col].nunique()}")
print(f"Missing values: {df_clean[target_col].isnull().sum()}")
print(f"Top value counts:")
print(df_clean[target_col].value_counts().head())

```

    
    Target variable 'Project_PRF_Normalised_Work_Effort' summary:
    Unique values: 4214
    Missing values: 0
    Top value counts:
    Project_PRF_Normalised_Work_Effort
    995    9
    304    9
    620    9
    473    9
    62     9
    Name: count, dtype: int64
    Cell executed at: 2025-05-16 16:40:43.194482
    


```python
# Check for infinite values
inf_check = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
print(f"\nNumber of infinite values: {inf_check}")
```

    
    Number of infinite values: 0
    Cell executed at: 2025-05-16 16:40:43.210289
    


```python
# Save cleaned data
df_clean.to_csv('data/cleaned_data.csv', index=False)
print("\nCleaned data saved to 'data/cleaned_data.csv'")

```

    
    Cleaned data saved to 'data/cleaned_data.csv'
    Cell executed at: 2025-05-16 16:40:43.386004
    

[Back to top](#Index:)

<a id='part3'></a>

# Part 3 - Feature Engineering and Selection

Involves creating or selecting specific features for the model based on insights from EDA, including handling categorical variables and reducing dimensionality if necessary.


```python
# Identify categorical columns and check cardinality
print("\nCategorical columns and their cardinality:")
cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols[:5]:  # Show first 5
    print(f"  {col}: {df_clean[col].nunique()} unique values")
if len(cat_cols) > 5:
    print(f"  ... and {len(cat_cols) - 5} more columns")
```

    
    Categorical columns and their cardinality:
      External_EEF_Data_Quality_Rating: 2 unique values
      External_EEF_Industry_Sector: 18 unique values
      External_EEF_Organisation_Type: 180 unique values
      Project_PRF_Application_Group: 7 unique values
      Project_PRF_Application_Type: 540 unique values
      ... and 11 more columns
    Cell executed at: 2025-05-16 16:40:43.409202
    


```python
# One-hot encode categorical columns with low cardinality (<10 unique values)
low_card_cols = [col for col in cat_cols if df_clean[col].nunique() < 10]
print(f"\nApplying one-hot encoding to {len(low_card_cols)} low-cardinality columns:")
for col in low_card_cols[:5]:  # Show first 5
    print(f"  - {col}")
if len(low_card_cols) > 5:
    print(f"  - ... and {len(low_card_cols) - 5} more columns")

```

    
    Applying one-hot encoding to 9 low-cardinality columns:
      - External_EEF_Data_Quality_Rating
      - Project_PRF_Application_Group
      - Project_PRF_Development_Type
      - Tech_TF_Development_Platform
      - Tech_TF_Language_Type
      - ... and 4 more columns
    Cell executed at: 2025-05-16 16:40:43.438545
    


```python
# Create encoded dataframe
df_encoded = pd.get_dummies(df_clean, columns=low_card_cols, drop_first=True)
print(f"\nData shape after one-hot encoding: {df_encoded.shape}")
print("\nAll column names:")
print(df_encoded.columns.tolist())

```

    
    Data shape after one-hot encoding: (7058, 56)
    
    All column names:
    ['ISBSG_Project_ID', 'Project_PRF_Year_of_Project', 'External_EEF_Industry_Sector', 'External_EEF_Organisation_Type', 'Project_PRF_Application_Type', 'Tech_TF_Primary_Programming_Language', 'Project_PRF_Functional_Size', 'Project_PRF_Relative_Size', 'Project_PRF_Normalised_Work_Effort_Level_1', 'Project_PRF_Normalised_Work_Effort', 'Project_PRF_Normalised_Level_1_PDR_ufp', 'Project_PRF_Normalised_PDR_ufp', 'Project_PRF_Speed_of_Delivery', 'Project_PRF_Project_Elapsed_Time', 'Project_PRF_Team_Size_Group', 'Project_PRF_Max_Team_Size', 'Process_PMF_Development_Methodologies', 'Process_PMF_Docs', 'Tech_TF_Tools_Used', 'External_EEF_Data_Quality_Rating_B', 'Project_PRF_Application_Group_Business Application; Infrastructure Software;', 'Project_PRF_Application_Group_Infrastructure Software', 'Project_PRF_Application_Group_Mathematically intensive application', 'Project_PRF_Application_Group_Mathematically-Intensive Application', 'Project_PRF_Application_Group_Missing', 'Project_PRF_Application_Group_Real-Time Application', 'Project_PRF_Development_Type_New Development', 'Project_PRF_Development_Type_Re-development', 'Tech_TF_Development_Platform_MF', 'Tech_TF_Development_Platform_MR', 'Tech_TF_Development_Platform_Missing', 'Tech_TF_Development_Platform_Multi', 'Tech_TF_Development_Platform_PC', 'Tech_TF_Development_Platform_Proprietary', 'Tech_TF_Language_Type_3GL', 'Tech_TF_Language_Type_4GL', 'Tech_TF_Language_Type_5GL', 'Tech_TF_Language_Type_APG', 'Tech_TF_Language_Type_ApG', 'Tech_TF_Language_Type_Missing', 'Project_PRF_CASE_Tool_Used_Missing', 'Project_PRF_CASE_Tool_Used_No', 'Project_PRF_CASE_Tool_Used_Yes', 'Tech_TF_Architecture_Missing', 'Tech_TF_Architecture_Multi-tier', 'Tech_TF_Architecture_Multi-tier / Client server', 'Tech_TF_Architecture_Multi-tier with web interface', 'Tech_TF_Architecture_Multi-tier with web public interface', 'Tech_TF_Architecture_Stand alone', 'Tech_TF_Architecture_Stand-alone', 'Tech_TF_Client_Server_Missing', 'Tech_TF_Client_Server_No', 'Tech_TF_Client_Server_Not Applicable', 'Tech_TF_Client_Server_Yes', 'Tech_TF_DBMS_Used_No', 'Tech_TF_DBMS_Used_Yes']
    Cell executed at: 2025-05-16 16:40:43.474521
    


```python
# MANUALLY fix the problematic column names BEFORE PyCaret setup

# Function to fix the column names and prevent duplicates
def fix_column_names_no_duplicates(df):
    """Fix column names that cause issues with PyCaret while preventing duplicates."""
    original_cols = df.columns.tolist()
    fixed_columns = []
    
    # Track columns to check for duplicates
    seen_columns = set()
    
    for col in original_cols:
        # Replace spaces with underscores
        fixed_col = col.replace(' ', '_')
        # Replace ampersands 
        fixed_col = fixed_col.replace('&', 'and')
        # Remove any other problematic characters
        fixed_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in fixed_col)
        
        # Handle duplicates by appending a suffix
        base_col = fixed_col
        suffix = 1
        while fixed_col in seen_columns:
            fixed_col = f"{base_col}_{suffix}"
            suffix += 1
        
        seen_columns.add(fixed_col)
        fixed_columns.append(fixed_col)
    
    # Create a new DataFrame with fixed column names
    df_fixed = df.copy()
    df_fixed.columns = fixed_columns
    
    # Print statistics about the renaming
    n_changed = sum(1 for old, new in zip(original_cols, fixed_columns) if old != new)
    print(f"Changed {n_changed} column names.")
    
    # Check for duplicates in the new column names
    dup_check = [item for item, count in pd.Series(fixed_columns).value_counts().items() if count > 1]
    if dup_check:
        print(f"WARNING: Found {len(dup_check)} duplicate column names after fixing: {dup_check}")
    else:
        print("No duplicate column names in the fixed DataFrame.")
    
    return df_fixed

# Show some of the original column names to help diagnose issues
print("\nSample of original column names:")
for i, col in enumerate(df_encoded.columns[:15]):  # Show first 15 for diagnosis
    print(f"{i}: {col}")

# Apply the fix to your dataframe
print("\nFixing column names for PyCaret compatibility...")
df_fixed = fix_column_names_no_duplicates(df_encoded)

# Print some example fixed columns to verify
print("\nSample of fixed column names:")
for i, (old, new) in enumerate(zip(df_encoded.columns[:15], df_fixed.columns[:15])):
    print(f"Original: {old} -> Fixed: {new}")
```

    
    Sample of original column names:
    0: ISBSG_Project_ID
    1: Project_PRF_Year_of_Project
    2: External_EEF_Industry_Sector
    3: External_EEF_Organisation_Type
    4: Project_PRF_Application_Type
    5: Tech_TF_Primary_Programming_Language
    6: Project_PRF_Functional_Size
    7: Project_PRF_Relative_Size
    8: Project_PRF_Normalised_Work_Effort_Level_1
    9: Project_PRF_Normalised_Work_Effort
    10: Project_PRF_Normalised_Level_1_PDR_ufp
    11: Project_PRF_Normalised_PDR_ufp
    12: Project_PRF_Speed_of_Delivery
    13: Project_PRF_Project_Elapsed_Time
    14: Project_PRF_Team_Size_Group
    
    Fixing column names for PyCaret compatibility...
    Changed 14 column names.
    No duplicate column names in the fixed DataFrame.
    
    Sample of fixed column names:
    Original: ISBSG_Project_ID -> Fixed: ISBSG_Project_ID
    Original: Project_PRF_Year_of_Project -> Fixed: Project_PRF_Year_of_Project
    Original: External_EEF_Industry_Sector -> Fixed: External_EEF_Industry_Sector
    Original: External_EEF_Organisation_Type -> Fixed: External_EEF_Organisation_Type
    Original: Project_PRF_Application_Type -> Fixed: Project_PRF_Application_Type
    Original: Tech_TF_Primary_Programming_Language -> Fixed: Tech_TF_Primary_Programming_Language
    Original: Project_PRF_Functional_Size -> Fixed: Project_PRF_Functional_Size
    Original: Project_PRF_Relative_Size -> Fixed: Project_PRF_Relative_Size
    Original: Project_PRF_Normalised_Work_Effort_Level_1 -> Fixed: Project_PRF_Normalised_Work_Effort_Level_1
    Original: Project_PRF_Normalised_Work_Effort -> Fixed: Project_PRF_Normalised_Work_Effort
    Original: Project_PRF_Normalised_Level_1_PDR_ufp -> Fixed: Project_PRF_Normalised_Level_1_PDR_ufp
    Original: Project_PRF_Normalised_PDR_ufp -> Fixed: Project_PRF_Normalised_PDR_ufp
    Original: Project_PRF_Speed_of_Delivery -> Fixed: Project_PRF_Speed_of_Delivery
    Original: Project_PRF_Project_Elapsed_Time -> Fixed: Project_PRF_Project_Elapsed_Time
    Original: Project_PRF_Team_Size_Group -> Fixed: Project_PRF_Team_Size_Group
    Cell executed at: 2025-05-16 16:40:43.506854
    


```python
# Save this DataFrame with fixed column names
df_fixed.to_csv('data/fixed_columns_data.csv', index=False)
print(f"Saved data with fixed column names to 'data/fixed_columns_data.csv'")
```

    Saved data with fixed column names to 'data/fixed_columns_data.csv'
    Cell executed at: 2025-05-16 16:40:43.725124
    


```python
# Create a diagnostic file with all column transformations
with open('column_transformations.txt', 'w') as f:
    f.write("Column name transformations:\n")
    f.write("--------------------------\n")
    for old, new in zip(df_encoded.columns, df_fixed.columns):
        f.write(f"{old} -> {new}\n")
print("Saved complete column transformations to 'column_transformations.txt'")
```

    Saved complete column transformations to 'column_transformations.txt'
    Cell executed at: 2025-05-16 16:40:43.741221
    

[Back to top](#Index:)

<a id='part4'></a>

# Part 4 - Data Profiling

xxx


```python
# ## Data Profiling (Optional)

try:
    from ydata_profiling import ProfileReport
    
    print("\nGenerating data profile report...")
    profile = ProfileReport(df_clean, title="ISBSG Dataset Profiling Report", minimal=True)
    profile.to_file("data_profile.html")
    print("Data profile report saved to 'data_profile.html'")
except ImportError:
    print("\nSkipping data profiling (ydata_profiling not installed)")
    print("To install: pip install ydata-profiling")
```



<div>
    <ins><a href="https://ydata.ai/register">Upgrade to ydata-sdk</a></ins>
    <p>
        Improve your data and profiling with ydata-sdk, featuring data quality scoring, redundancy detection, outlier identification, text validation, and synthetic data generation.
    </p>
</div>



    
    Generating data profile report...
    


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]


    
    [A%|          | 0/28 [00:00<?, ?it/s]
    [A%|█▍        | 4/28 [00:00<00:00, 36.90it/s]
    100%|██████████| 28/28 [00:00<00:00, 61.49it/s]
    


    Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]



    Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]



    Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]


    Data profile report saved to 'data_profile.html'
    Cell executed at: 2025-05-16 16:41:10.837847
    

[Back to top](#Index:)

<a id='part5'></a>

# Part 5 - Model Building with PyCaret

xxx


```python
# Import PyCaret regression module
from pycaret.regression import setup, compare_models, create_model, pull, plot_model
from pycaret.regression import tune_model, evaluate_model, save_model, get_config

# Setup PyCaret environment ONCE
print("\nSetting up PyCaret environment...")
try:
    # Split data into features and target
    target_col_fixed = target_col  # Adjust if your target column name changed
    X = df_fixed.drop(columns=[target_col_fixed])
    y = df_fixed[target_col_fixed]
    
    # PyCaret setup with MINIMAL preprocessing
    # The key is to avoid PyCaret's automatic preprocessing of column names
    setup_results = setup(
        data=df_fixed,
        target=target_col_fixed,
        session_id=123,
        preprocess=True,
        verbose=False
    )
except Exception as e:
    print("Error during PyCaret setup:", e)
```

    
    Setting up PyCaret environment...
    Cell executed at: 2025-05-16 16:41:28.731424
    


```python
    # Get preprocessed data for later use
    pycaret_X = get_config("X")
    pycaret_y = get_config("y")
    
    # Check data info
    print(f"\nPreprocessed data shape: {pycaret_X.shape}")
    print(f"Numeric features: {len(pycaret_X.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical features: {len(pycaret_X.select_dtypes(include=['object', 'category']).columns)}")
    
    # Save preprocessed data
    pycaret_X.to_csv('data/pycaret_processed_features.csv', index=False)
    pycaret_y.to_csv('data/pycaret_processed_target.csv', index=False)
    print("PyCaret preprocessed data saved to CSV files")

```

    
    Preprocessed data shape: (7058, 55)
    Numeric features: 11
    Categorical features: 7
    PyCaret preprocessed data saved to CSV files
    Cell executed at: 2025-05-16 16:41:28.990953
    


```python
   
# Compare regression models
print("\nComparing regression models...")
best_models = compare_models(n_select=3)  # Select top 3 models
model_results = pull()
print("\nModel comparison results:")
print(model_results)
```

    
    Comparing regression models...
    






<style type="text/css">
#T_94f5a th {
  text-align: left;
}
#T_94f5a_row0_col0, #T_94f5a_row1_col0, #T_94f5a_row1_col1, #T_94f5a_row1_col2, #T_94f5a_row1_col3, #T_94f5a_row1_col4, #T_94f5a_row1_col5, #T_94f5a_row1_col6, #T_94f5a_row2_col0, #T_94f5a_row2_col1, #T_94f5a_row2_col2, #T_94f5a_row2_col3, #T_94f5a_row2_col4, #T_94f5a_row2_col5, #T_94f5a_row2_col6, #T_94f5a_row3_col0, #T_94f5a_row3_col1, #T_94f5a_row3_col2, #T_94f5a_row3_col3, #T_94f5a_row3_col4, #T_94f5a_row3_col5, #T_94f5a_row3_col6, #T_94f5a_row4_col0, #T_94f5a_row4_col1, #T_94f5a_row4_col2, #T_94f5a_row4_col3, #T_94f5a_row4_col4, #T_94f5a_row4_col5, #T_94f5a_row4_col6, #T_94f5a_row5_col0, #T_94f5a_row5_col1, #T_94f5a_row5_col2, #T_94f5a_row5_col3, #T_94f5a_row5_col4, #T_94f5a_row5_col5, #T_94f5a_row5_col6, #T_94f5a_row6_col0, #T_94f5a_row6_col1, #T_94f5a_row6_col2, #T_94f5a_row6_col3, #T_94f5a_row6_col4, #T_94f5a_row6_col5, #T_94f5a_row6_col6, #T_94f5a_row7_col0, #T_94f5a_row7_col1, #T_94f5a_row7_col2, #T_94f5a_row7_col3, #T_94f5a_row7_col4, #T_94f5a_row7_col5, #T_94f5a_row7_col6, #T_94f5a_row8_col0, #T_94f5a_row8_col1, #T_94f5a_row8_col2, #T_94f5a_row8_col3, #T_94f5a_row8_col4, #T_94f5a_row8_col5, #T_94f5a_row8_col6, #T_94f5a_row9_col0, #T_94f5a_row9_col1, #T_94f5a_row9_col2, #T_94f5a_row9_col3, #T_94f5a_row9_col4, #T_94f5a_row9_col5, #T_94f5a_row9_col6, #T_94f5a_row10_col0, #T_94f5a_row10_col1, #T_94f5a_row10_col2, #T_94f5a_row10_col3, #T_94f5a_row10_col4, #T_94f5a_row10_col5, #T_94f5a_row10_col6, #T_94f5a_row11_col0, #T_94f5a_row11_col1, #T_94f5a_row11_col2, #T_94f5a_row11_col3, #T_94f5a_row11_col4, #T_94f5a_row11_col5, #T_94f5a_row11_col6, #T_94f5a_row12_col0, #T_94f5a_row12_col1, #T_94f5a_row12_col2, #T_94f5a_row12_col3, #T_94f5a_row12_col4, #T_94f5a_row12_col5, #T_94f5a_row12_col6, #T_94f5a_row13_col0, #T_94f5a_row13_col1, #T_94f5a_row13_col2, #T_94f5a_row13_col3, #T_94f5a_row13_col4, #T_94f5a_row13_col5, #T_94f5a_row13_col6, #T_94f5a_row14_col0, #T_94f5a_row14_col1, #T_94f5a_row14_col2, #T_94f5a_row14_col3, #T_94f5a_row14_col4, #T_94f5a_row14_col5, #T_94f5a_row14_col6, #T_94f5a_row15_col0, #T_94f5a_row15_col1, #T_94f5a_row15_col2, #T_94f5a_row15_col3, #T_94f5a_row15_col4, #T_94f5a_row15_col5, #T_94f5a_row15_col6, #T_94f5a_row16_col0, #T_94f5a_row16_col1, #T_94f5a_row16_col2, #T_94f5a_row16_col3, #T_94f5a_row16_col4, #T_94f5a_row16_col5, #T_94f5a_row16_col6, #T_94f5a_row17_col0, #T_94f5a_row17_col1, #T_94f5a_row17_col2, #T_94f5a_row17_col3, #T_94f5a_row17_col4, #T_94f5a_row17_col5, #T_94f5a_row17_col6, #T_94f5a_row18_col0, #T_94f5a_row18_col1, #T_94f5a_row18_col2, #T_94f5a_row18_col3, #T_94f5a_row18_col4, #T_94f5a_row18_col5, #T_94f5a_row18_col6, #T_94f5a_row19_col0, #T_94f5a_row19_col1, #T_94f5a_row19_col2, #T_94f5a_row19_col3, #T_94f5a_row19_col4, #T_94f5a_row19_col5, #T_94f5a_row19_col6 {
  text-align: left;
}
#T_94f5a_row0_col1, #T_94f5a_row0_col2, #T_94f5a_row0_col3, #T_94f5a_row0_col4, #T_94f5a_row0_col5, #T_94f5a_row0_col6 {
  text-align: left;
  background-color: yellow;
}
#T_94f5a_row0_col7, #T_94f5a_row1_col7, #T_94f5a_row2_col7, #T_94f5a_row3_col7, #T_94f5a_row4_col7, #T_94f5a_row5_col7, #T_94f5a_row6_col7, #T_94f5a_row7_col7, #T_94f5a_row8_col7, #T_94f5a_row9_col7, #T_94f5a_row10_col7, #T_94f5a_row11_col7, #T_94f5a_row12_col7, #T_94f5a_row13_col7, #T_94f5a_row14_col7, #T_94f5a_row15_col7, #T_94f5a_row16_col7, #T_94f5a_row17_col7, #T_94f5a_row19_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_94f5a_row18_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_94f5a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_94f5a_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_94f5a_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_94f5a_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_94f5a_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_94f5a_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_94f5a_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_94f5a_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_94f5a_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_94f5a_level0_row0" class="row_heading level0 row0" >et</th>
      <td id="T_94f5a_row0_col0" class="data row0 col0" >Extra Trees Regressor</td>
      <td id="T_94f5a_row0_col1" class="data row0 col1" >503.2762</td>
      <td id="T_94f5a_row0_col2" class="data row0 col2" >13277409.6465</td>
      <td id="T_94f5a_row0_col3" class="data row0 col3" >3167.2358</td>
      <td id="T_94f5a_row0_col4" class="data row0 col4" >0.8926</td>
      <td id="T_94f5a_row0_col5" class="data row0 col5" >0.2069</td>
      <td id="T_94f5a_row0_col6" class="data row0 col6" >0.0995</td>
      <td id="T_94f5a_row0_col7" class="data row0 col7" >2.1700</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row1" class="row_heading level0 row1" >gbr</th>
      <td id="T_94f5a_row1_col0" class="data row1 col0" >Gradient Boosting Regressor</td>
      <td id="T_94f5a_row1_col1" class="data row1 col1" >719.3396</td>
      <td id="T_94f5a_row1_col2" class="data row1 col2" >14149244.9609</td>
      <td id="T_94f5a_row1_col3" class="data row1 col3" >3290.5400</td>
      <td id="T_94f5a_row1_col4" class="data row1 col4" >0.8878</td>
      <td id="T_94f5a_row1_col5" class="data row1 col5" >0.5085</td>
      <td id="T_94f5a_row1_col6" class="data row1 col6" >0.5309</td>
      <td id="T_94f5a_row1_col7" class="data row1 col7" >0.7250</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row2" class="row_heading level0 row2" >rf</th>
      <td id="T_94f5a_row2_col0" class="data row2 col0" >Random Forest Regressor</td>
      <td id="T_94f5a_row2_col1" class="data row2 col1" >525.4149</td>
      <td id="T_94f5a_row2_col2" class="data row2 col2" >13565340.0642</td>
      <td id="T_94f5a_row2_col3" class="data row2 col3" >3294.8602</td>
      <td id="T_94f5a_row2_col4" class="data row2 col4" >0.8849</td>
      <td id="T_94f5a_row2_col5" class="data row2 col5" >0.2184</td>
      <td id="T_94f5a_row2_col6" class="data row2 col6" >0.1034</td>
      <td id="T_94f5a_row2_col7" class="data row2 col7" >2.2520</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row3" class="row_heading level0 row3" >xgboost</th>
      <td id="T_94f5a_row3_col0" class="data row3 col0" >Extreme Gradient Boosting</td>
      <td id="T_94f5a_row3_col1" class="data row3 col1" >604.2027</td>
      <td id="T_94f5a_row3_col2" class="data row3 col2" >16188296.5906</td>
      <td id="T_94f5a_row3_col3" class="data row3 col3" >3276.8846</td>
      <td id="T_94f5a_row3_col4" class="data row3 col4" >0.8830</td>
      <td id="T_94f5a_row3_col5" class="data row3 col5" >0.3133</td>
      <td id="T_94f5a_row3_col6" class="data row3 col6" >0.1771</td>
      <td id="T_94f5a_row3_col7" class="data row3 col7" >0.2440</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row4" class="row_heading level0 row4" >catboost</th>
      <td id="T_94f5a_row4_col0" class="data row4 col0" >CatBoost Regressor</td>
      <td id="T_94f5a_row4_col1" class="data row4 col1" >600.4510</td>
      <td id="T_94f5a_row4_col2" class="data row4 col2" >19946043.8889</td>
      <td id="T_94f5a_row4_col3" class="data row4 col3" >3689.9275</td>
      <td id="T_94f5a_row4_col4" class="data row4 col4" >0.8587</td>
      <td id="T_94f5a_row4_col5" class="data row4 col5" >0.3837</td>
      <td id="T_94f5a_row4_col6" class="data row4 col6" >0.2769</td>
      <td id="T_94f5a_row4_col7" class="data row4 col7" >2.9350</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row5" class="row_heading level0 row5" >lightgbm</th>
      <td id="T_94f5a_row5_col0" class="data row5 col0" >Light Gradient Boosting Machine</td>
      <td id="T_94f5a_row5_col1" class="data row5 col1" >810.1306</td>
      <td id="T_94f5a_row5_col2" class="data row5 col2" >26476195.6065</td>
      <td id="T_94f5a_row5_col3" class="data row5 col3" >4513.4841</td>
      <td id="T_94f5a_row5_col4" class="data row5 col4" >0.7957</td>
      <td id="T_94f5a_row5_col5" class="data row5 col5" >0.4070</td>
      <td id="T_94f5a_row5_col6" class="data row5 col6" >0.3129</td>
      <td id="T_94f5a_row5_col7" class="data row5 col7" >0.3110</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row6" class="row_heading level0 row6" >dt</th>
      <td id="T_94f5a_row6_col0" class="data row6 col0" >Decision Tree Regressor</td>
      <td id="T_94f5a_row6_col1" class="data row6 col1" >638.0895</td>
      <td id="T_94f5a_row6_col2" class="data row6 col2" >24073135.6644</td>
      <td id="T_94f5a_row6_col3" class="data row6 col3" >4336.3695</td>
      <td id="T_94f5a_row6_col4" class="data row6 col4" >0.7778</td>
      <td id="T_94f5a_row6_col5" class="data row6 col5" >0.2453</td>
      <td id="T_94f5a_row6_col6" class="data row6 col6" >0.1016</td>
      <td id="T_94f5a_row6_col7" class="data row6 col7" >0.1260</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row7" class="row_heading level0 row7" >lasso</th>
      <td id="T_94f5a_row7_col0" class="data row7 col0" >Lasso Regression</td>
      <td id="T_94f5a_row7_col1" class="data row7 col1" >1671.7410</td>
      <td id="T_94f5a_row7_col2" class="data row7 col2" >24874465.7860</td>
      <td id="T_94f5a_row7_col3" class="data row7 col3" >4561.1952</td>
      <td id="T_94f5a_row7_col4" class="data row7 col4" >0.7643</td>
      <td id="T_94f5a_row7_col5" class="data row7 col5" >1.0610</td>
      <td id="T_94f5a_row7_col6" class="data row7 col6" >1.9398</td>
      <td id="T_94f5a_row7_col7" class="data row7 col7" >0.3060</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row8" class="row_heading level0 row8" >ridge</th>
      <td id="T_94f5a_row8_col0" class="data row8 col0" >Ridge Regression</td>
      <td id="T_94f5a_row8_col1" class="data row8 col1" >1679.2914</td>
      <td id="T_94f5a_row8_col2" class="data row8 col2" >24863359.4279</td>
      <td id="T_94f5a_row8_col3" class="data row8 col3" >4560.7502</td>
      <td id="T_94f5a_row8_col4" class="data row8 col4" >0.7643</td>
      <td id="T_94f5a_row8_col5" class="data row8 col5" >1.0462</td>
      <td id="T_94f5a_row8_col6" class="data row8 col6" >1.9424</td>
      <td id="T_94f5a_row8_col7" class="data row8 col7" >0.0900</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row9" class="row_heading level0 row9" >llar</th>
      <td id="T_94f5a_row9_col0" class="data row9 col0" >Lasso Least Angle Regression</td>
      <td id="T_94f5a_row9_col1" class="data row9 col1" >1672.0987</td>
      <td id="T_94f5a_row9_col2" class="data row9 col2" >24874684.3109</td>
      <td id="T_94f5a_row9_col3" class="data row9 col3" >4561.2921</td>
      <td id="T_94f5a_row9_col4" class="data row9 col4" >0.7643</td>
      <td id="T_94f5a_row9_col5" class="data row9 col5" >1.0602</td>
      <td id="T_94f5a_row9_col6" class="data row9 col6" >1.9403</td>
      <td id="T_94f5a_row9_col7" class="data row9 col7" >0.1060</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row10" class="row_heading level0 row10" >en</th>
      <td id="T_94f5a_row10_col0" class="data row10 col0" >Elastic Net</td>
      <td id="T_94f5a_row10_col1" class="data row10 col1" >1433.1870</td>
      <td id="T_94f5a_row10_col2" class="data row10 col2" >25538394.5968</td>
      <td id="T_94f5a_row10_col3" class="data row10 col3" >4591.0118</td>
      <td id="T_94f5a_row10_col4" class="data row10 col4" >0.7637</td>
      <td id="T_94f5a_row10_col5" class="data row10 col5" >0.9390</td>
      <td id="T_94f5a_row10_col6" class="data row10 col6" >1.5949</td>
      <td id="T_94f5a_row10_col7" class="data row10 col7" >0.3090</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row11" class="row_heading level0 row11" >lr</th>
      <td id="T_94f5a_row11_col0" class="data row11 col0" >Linear Regression</td>
      <td id="T_94f5a_row11_col1" class="data row11 col1" >1697.8919</td>
      <td id="T_94f5a_row11_col2" class="data row11 col2" >24912887.8911</td>
      <td id="T_94f5a_row11_col3" class="data row11 col3" >4569.8871</td>
      <td id="T_94f5a_row11_col4" class="data row11 col4" >0.7633</td>
      <td id="T_94f5a_row11_col5" class="data row11 col5" >1.0567</td>
      <td id="T_94f5a_row11_col6" class="data row11 col6" >1.9451</td>
      <td id="T_94f5a_row11_col7" class="data row11 col7" >1.6260</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row12" class="row_heading level0 row12" >br</th>
      <td id="T_94f5a_row12_col0" class="data row12 col0" >Bayesian Ridge</td>
      <td id="T_94f5a_row12_col1" class="data row12 col1" >1508.6388</td>
      <td id="T_94f5a_row12_col2" class="data row12 col2" >26956133.1110</td>
      <td id="T_94f5a_row12_col3" class="data row12 col3" >4764.5869</td>
      <td id="T_94f5a_row12_col4" class="data row12 col4" >0.7464</td>
      <td id="T_94f5a_row12_col5" class="data row12 col5" >1.0096</td>
      <td id="T_94f5a_row12_col6" class="data row12 col6" >1.8624</td>
      <td id="T_94f5a_row12_col7" class="data row12 col7" >0.1130</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row13" class="row_heading level0 row13" >omp</th>
      <td id="T_94f5a_row13_col0" class="data row13 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_94f5a_row13_col1" class="data row13 col1" >1506.7213</td>
      <td id="T_94f5a_row13_col2" class="data row13 col2" >27009348.6692</td>
      <td id="T_94f5a_row13_col3" class="data row13 col3" >4769.3113</td>
      <td id="T_94f5a_row13_col4" class="data row13 col4" >0.7460</td>
      <td id="T_94f5a_row13_col5" class="data row13 col5" >1.0072</td>
      <td id="T_94f5a_row13_col6" class="data row13 col6" >1.8322</td>
      <td id="T_94f5a_row13_col7" class="data row13 col7" >0.0940</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row14" class="row_heading level0 row14" >huber</th>
      <td id="T_94f5a_row14_col0" class="data row14 col0" >Huber Regressor</td>
      <td id="T_94f5a_row14_col1" class="data row14 col1" >806.0686</td>
      <td id="T_94f5a_row14_col2" class="data row14 col2" >27876435.8843</td>
      <td id="T_94f5a_row14_col3" class="data row14 col3" >4819.8868</td>
      <td id="T_94f5a_row14_col4" class="data row14 col4" >0.7364</td>
      <td id="T_94f5a_row14_col5" class="data row14 col5" >0.5005</td>
      <td id="T_94f5a_row14_col6" class="data row14 col6" >0.3414</td>
      <td id="T_94f5a_row14_col7" class="data row14 col7" >0.2740</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row15" class="row_heading level0 row15" >par</th>
      <td id="T_94f5a_row15_col0" class="data row15 col0" >Passive Aggressive Regressor</td>
      <td id="T_94f5a_row15_col1" class="data row15 col1" >1458.1774</td>
      <td id="T_94f5a_row15_col2" class="data row15 col2" >27780359.7492</td>
      <td id="T_94f5a_row15_col3" class="data row15 col3" >4912.6839</td>
      <td id="T_94f5a_row15_col4" class="data row15 col4" >0.7353</td>
      <td id="T_94f5a_row15_col5" class="data row15 col5" >0.9556</td>
      <td id="T_94f5a_row15_col6" class="data row15 col6" >1.5092</td>
      <td id="T_94f5a_row15_col7" class="data row15 col7" >0.0940</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row16" class="row_heading level0 row16" >knn</th>
      <td id="T_94f5a_row16_col0" class="data row16 col0" >K Neighbors Regressor</td>
      <td id="T_94f5a_row16_col1" class="data row16 col1" >1223.1884</td>
      <td id="T_94f5a_row16_col2" class="data row16 col2" >29379214.0024</td>
      <td id="T_94f5a_row16_col3" class="data row16 col3" >5038.7925</td>
      <td id="T_94f5a_row16_col4" class="data row16 col4" >0.7286</td>
      <td id="T_94f5a_row16_col5" class="data row16 col5" >0.7151</td>
      <td id="T_94f5a_row16_col6" class="data row16 col6" >0.9260</td>
      <td id="T_94f5a_row16_col7" class="data row16 col7" >0.1020</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row17" class="row_heading level0 row17" >ada</th>
      <td id="T_94f5a_row17_col0" class="data row17 col0" >AdaBoost Regressor</td>
      <td id="T_94f5a_row17_col1" class="data row17 col1" >4069.0582</td>
      <td id="T_94f5a_row17_col2" class="data row17 col2" >40171987.5841</td>
      <td id="T_94f5a_row17_col3" class="data row17 col3" >6022.6857</td>
      <td id="T_94f5a_row17_col4" class="data row17 col4" >0.5969</td>
      <td id="T_94f5a_row17_col5" class="data row17 col5" >1.6909</td>
      <td id="T_94f5a_row17_col6" class="data row17 col6" >8.5051</td>
      <td id="T_94f5a_row17_col7" class="data row17 col7" >0.3540</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row18" class="row_heading level0 row18" >dummy</th>
      <td id="T_94f5a_row18_col0" class="data row18 col0" >Dummy Regressor</td>
      <td id="T_94f5a_row18_col1" class="data row18 col1" >4394.2583</td>
      <td id="T_94f5a_row18_col2" class="data row18 col2" >101895046.8525</td>
      <td id="T_94f5a_row18_col3" class="data row18 col3" >9748.0736</td>
      <td id="T_94f5a_row18_col4" class="data row18 col4" >-0.0025</td>
      <td id="T_94f5a_row18_col5" class="data row18 col5" >1.7383</td>
      <td id="T_94f5a_row18_col6" class="data row18 col6" >8.1066</td>
      <td id="T_94f5a_row18_col7" class="data row18 col7" >0.0820</td>
    </tr>
    <tr>
      <th id="T_94f5a_level0_row19" class="row_heading level0 row19" >lar</th>
      <td id="T_94f5a_row19_col0" class="data row19 col0" >Least Angle Regression</td>
      <td id="T_94f5a_row19_col1" class="data row19 col1" >9904132210027578343606089366560740322910362679911414370288952667813912094680691619290747802017937354394776539148647282898303325479214119965505128918252586621693299547794458059362725272772770234137480334302255310183298367058832792757710075824942196537086580279067665235968.0000</td>
      <td id="T_94f5a_row19_col2" class="data row19 col2" >inf</td>
      <td id="T_94f5a_row19_col3" class="data row19 col3" >inf</td>
      <td id="T_94f5a_row19_col4" class="data row19 col4" >-inf</td>
      <td id="T_94f5a_row19_col5" class="data row19 col5" >66.6141</td>
      <td id="T_94f5a_row19_col6" class="data row19 col6" >23023381162017265276406949564882731659548520113073729290229311420285928854274498363551572754644727087724423912997019759399329216016532409099489521916022772336367681401969984279399146681522892969915442547518195645754300351542822291856130143891632315029084462964619083776.0000</td>
      <td id="T_94f5a_row19_col7" class="data row19 col7" >0.1160</td>
    </tr>
  </tbody>
</table>







    
    Model comparison results:
                                        Model            MAE           MSE  \
    et                  Extra Trees Regressor   5.032762e+02  1.327741e+07   
    gbr           Gradient Boosting Regressor   7.193396e+02  1.414924e+07   
    rf                Random Forest Regressor   5.254149e+02  1.356534e+07   
    xgboost         Extreme Gradient Boosting   6.042027e+02  1.618830e+07   
    catboost               CatBoost Regressor   6.004510e+02  1.994604e+07   
    lightgbm  Light Gradient Boosting Machine   8.101306e+02  2.647620e+07   
    dt                Decision Tree Regressor   6.380895e+02  2.407314e+07   
    lasso                    Lasso Regression   1.671741e+03  2.487447e+07   
    ridge                    Ridge Regression   1.679291e+03  2.486336e+07   
    llar         Lasso Least Angle Regression   1.672099e+03  2.487468e+07   
    en                            Elastic Net   1.433187e+03  2.553839e+07   
    lr                      Linear Regression   1.697892e+03  2.491289e+07   
    br                         Bayesian Ridge   1.508639e+03  2.695613e+07   
    omp           Orthogonal Matching Pursuit   1.506721e+03  2.700935e+07   
    huber                     Huber Regressor   8.060686e+02  2.787644e+07   
    par          Passive Aggressive Regressor   1.458177e+03  2.778036e+07   
    knn                 K Neighbors Regressor   1.223188e+03  2.937921e+07   
    ada                    AdaBoost Regressor   4.069058e+03  4.017199e+07   
    dummy                     Dummy Regressor   4.394258e+03  1.018950e+08   
    lar                Least Angle Regression  9.904132e+270           inf   
    
                   RMSE      R2    RMSLE           MAPE  TT (Sec)  
    et        3167.2358  0.8926   0.2069   9.950000e-02     2.170  
    gbr       3290.5400  0.8878   0.5085   5.309000e-01     0.725  
    rf        3294.8602  0.8849   0.2184   1.034000e-01     2.252  
    xgboost   3276.8846  0.8830   0.3133   1.771000e-01     0.244  
    catboost  3689.9275  0.8587   0.3837   2.769000e-01     2.935  
    lightgbm  4513.4841  0.7957   0.4070   3.129000e-01     0.311  
    dt        4336.3695  0.7778   0.2453   1.016000e-01     0.126  
    lasso     4561.1952  0.7643   1.0610   1.939800e+00     0.306  
    ridge     4560.7502  0.7643   1.0462   1.942400e+00     0.090  
    llar      4561.2921  0.7643   1.0602   1.940300e+00     0.106  
    en        4591.0118  0.7637   0.9390   1.594900e+00     0.309  
    lr        4569.8871  0.7633   1.0567   1.945100e+00     1.626  
    br        4764.5869  0.7464   1.0096   1.862400e+00     0.113  
    omp       4769.3113  0.7460   1.0072   1.832200e+00     0.094  
    huber     4819.8868  0.7364   0.5005   3.414000e-01     0.274  
    par       4912.6839  0.7353   0.9556   1.509200e+00     0.094  
    knn       5038.7925  0.7286   0.7151   9.260000e-01     0.102  
    ada       6022.6857  0.5969   1.6909   8.505100e+00     0.354  
    dummy     9748.0736 -0.0025   1.7383   8.106600e+00     0.082  
    lar             inf    -inf  66.6141  2.302338e+268     0.116  
    Cell executed at: 2025-05-16 16:43:48.047022
    


```python
    # Select best model and create it
    best_model_name = model_results.index[0]
    print(f"\nCreating best model: {best_model_name}")
    model = create_model(best_model_name)
```

    
    Creating best model: et
    






<style type="text/css">
#T_f025b_row10_col0, #T_f025b_row10_col1, #T_f025b_row10_col2, #T_f025b_row10_col3, #T_f025b_row10_col4, #T_f025b_row10_col5 {
  background: yellow;
}
</style>
<table id="T_f025b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f025b_level0_col0" class="col_heading level0 col0" >MAE</th>
      <th id="T_f025b_level0_col1" class="col_heading level0 col1" >MSE</th>
      <th id="T_f025b_level0_col2" class="col_heading level0 col2" >RMSE</th>
      <th id="T_f025b_level0_col3" class="col_heading level0 col3" >R2</th>
      <th id="T_f025b_level0_col4" class="col_heading level0 col4" >RMSLE</th>
      <th id="T_f025b_level0_col5" class="col_heading level0 col5" >MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f025b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f025b_row0_col0" class="data row0 col0" >884.2739</td>
      <td id="T_f025b_row0_col1" class="data row0 col1" >55407139.4678</td>
      <td id="T_f025b_row0_col2" class="data row0 col2" >7443.5972</td>
      <td id="T_f025b_row0_col3" class="data row0 col3" >0.7730</td>
      <td id="T_f025b_row0_col4" class="data row0 col4" >0.2211</td>
      <td id="T_f025b_row0_col5" class="data row0 col5" >0.1014</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f025b_row1_col0" class="data row1 col0" >382.6128</td>
      <td id="T_f025b_row1_col1" class="data row1 col1" >2239050.3146</td>
      <td id="T_f025b_row1_col2" class="data row1 col2" >1496.3457</td>
      <td id="T_f025b_row1_col3" class="data row1 col3" >0.9640</td>
      <td id="T_f025b_row1_col4" class="data row1 col4" >0.1869</td>
      <td id="T_f025b_row1_col5" class="data row1 col5" >0.0881</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f025b_row2_col0" class="data row2 col0" >538.6472</td>
      <td id="T_f025b_row2_col1" class="data row2 col1" >6717338.7147</td>
      <td id="T_f025b_row2_col2" class="data row2 col2" >2591.7829</td>
      <td id="T_f025b_row2_col3" class="data row2 col3" >0.9261</td>
      <td id="T_f025b_row2_col4" class="data row2 col4" >0.2282</td>
      <td id="T_f025b_row2_col5" class="data row2 col5" >0.1214</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f025b_row3_col0" class="data row3 col0" >493.0081</td>
      <td id="T_f025b_row3_col1" class="data row3 col1" >9595418.2138</td>
      <td id="T_f025b_row3_col2" class="data row3 col2" >3097.6472</td>
      <td id="T_f025b_row3_col3" class="data row3 col3" >0.9346</td>
      <td id="T_f025b_row3_col4" class="data row3 col4" >0.1914</td>
      <td id="T_f025b_row3_col5" class="data row3 col5" >0.0818</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f025b_row4_col0" class="data row4 col0" >532.4317</td>
      <td id="T_f025b_row4_col1" class="data row4 col1" >24509494.4561</td>
      <td id="T_f025b_row4_col2" class="data row4 col2" >4950.7065</td>
      <td id="T_f025b_row4_col3" class="data row4 col3" >0.7689</td>
      <td id="T_f025b_row4_col4" class="data row4 col4" >0.1924</td>
      <td id="T_f025b_row4_col5" class="data row4 col5" >0.0832</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f025b_row5_col0" class="data row5 col0" >576.9346</td>
      <td id="T_f025b_row5_col1" class="data row5 col1" >17555614.6916</td>
      <td id="T_f025b_row5_col2" class="data row5 col2" >4189.9421</td>
      <td id="T_f025b_row5_col3" class="data row5 col3" >0.8284</td>
      <td id="T_f025b_row5_col4" class="data row5 col4" >0.2035</td>
      <td id="T_f025b_row5_col5" class="data row5 col5" >0.0708</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f025b_row6_col0" class="data row6 col0" >274.3069</td>
      <td id="T_f025b_row6_col1" class="data row6 col1" >1216929.0077</td>
      <td id="T_f025b_row6_col2" class="data row6 col2" >1103.1451</td>
      <td id="T_f025b_row6_col3" class="data row6 col3" >0.9647</td>
      <td id="T_f025b_row6_col4" class="data row6 col4" >0.1921</td>
      <td id="T_f025b_row6_col5" class="data row6 col5" >0.0986</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f025b_row7_col0" class="data row7 col0" >452.8986</td>
      <td id="T_f025b_row7_col1" class="data row7 col1" >3922056.4356</td>
      <td id="T_f025b_row7_col2" class="data row7 col2" >1980.4182</td>
      <td id="T_f025b_row7_col3" class="data row7 col3" >0.9656</td>
      <td id="T_f025b_row7_col4" class="data row7 col4" >0.2631</td>
      <td id="T_f025b_row7_col5" class="data row7 col5" >0.1565</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f025b_row8_col0" class="data row8 col0" >496.6227</td>
      <td id="T_f025b_row8_col1" class="data row8 col1" >5711179.1488</td>
      <td id="T_f025b_row8_col2" class="data row8 col2" >2389.8073</td>
      <td id="T_f025b_row8_col3" class="data row8 col3" >0.9035</td>
      <td id="T_f025b_row8_col4" class="data row8 col4" >0.2383</td>
      <td id="T_f025b_row8_col5" class="data row8 col5" >0.1239</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f025b_row9_col0" class="data row9 col0" >401.0256</td>
      <td id="T_f025b_row9_col1" class="data row9 col1" >5899876.0142</td>
      <td id="T_f025b_row9_col2" class="data row9 col2" >2428.9660</td>
      <td id="T_f025b_row9_col3" class="data row9 col3" >0.8970</td>
      <td id="T_f025b_row9_col4" class="data row9 col4" >0.1524</td>
      <td id="T_f025b_row9_col5" class="data row9 col5" >0.0696</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_f025b_row10_col0" class="data row10 col0" >503.2762</td>
      <td id="T_f025b_row10_col1" class="data row10 col1" >13277409.6465</td>
      <td id="T_f025b_row10_col2" class="data row10 col2" >3167.2358</td>
      <td id="T_f025b_row10_col3" class="data row10 col3" >0.8926</td>
      <td id="T_f025b_row10_col4" class="data row10 col4" >0.2069</td>
      <td id="T_f025b_row10_col5" class="data row10 col5" >0.0995</td>
    </tr>
    <tr>
      <th id="T_f025b_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_f025b_row11_col0" class="data row11 col0" >152.5829</td>
      <td id="T_f025b_row11_col1" class="data row11 col1" >15628395.7720</td>
      <td id="T_f025b_row11_col2" class="data row11 col2" >1801.6734</td>
      <td id="T_f025b_row11_col3" class="data row11 col3" >0.0724</td>
      <td id="T_f025b_row11_col4" class="data row11 col4" >0.0297</td>
      <td id="T_f025b_row11_col5" class="data row11 col5" >0.0260</td>
    </tr>
  </tbody>
</table>







    Cell executed at: 2025-05-16 17:01:14.428424
    


```python
    # Tune the best model
    print("\nTuning the best model...")
    tuned_model = tune_model(model, n_iter=10)
    
```

    
    Tuning the best model...
    






<style type="text/css">
#T_89860_row10_col0, #T_89860_row10_col1, #T_89860_row10_col2, #T_89860_row10_col3, #T_89860_row10_col4, #T_89860_row10_col5 {
  background: yellow;
}
</style>
<table id="T_89860">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_89860_level0_col0" class="col_heading level0 col0" >MAE</th>
      <th id="T_89860_level0_col1" class="col_heading level0 col1" >MSE</th>
      <th id="T_89860_level0_col2" class="col_heading level0 col2" >RMSE</th>
      <th id="T_89860_level0_col3" class="col_heading level0 col3" >R2</th>
      <th id="T_89860_level0_col4" class="col_heading level0 col4" >RMSLE</th>
      <th id="T_89860_level0_col5" class="col_heading level0 col5" >MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_89860_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_89860_row0_col0" class="data row0 col0" >1389.2786</td>
      <td id="T_89860_row0_col1" class="data row0 col1" >97825996.9469</td>
      <td id="T_89860_row0_col2" class="data row0 col2" >9890.7026</td>
      <td id="T_89860_row0_col3" class="data row0 col3" >0.5993</td>
      <td id="T_89860_row0_col4" class="data row0 col4" >0.6097</td>
      <td id="T_89860_row0_col5" class="data row0 col5" >0.9303</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_89860_row1_col0" class="data row1 col0" >905.6087</td>
      <td id="T_89860_row1_col1" class="data row1 col1" >8245823.2851</td>
      <td id="T_89860_row1_col2" class="data row1 col2" >2871.5542</td>
      <td id="T_89860_row1_col3" class="data row1 col3" >0.8674</td>
      <td id="T_89860_row1_col4" class="data row1 col4" >0.6199</td>
      <td id="T_89860_row1_col5" class="data row1 col5" >0.7719</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_89860_row2_col0" class="data row2 col0" >1032.6248</td>
      <td id="T_89860_row2_col1" class="data row2 col1" >18340253.6999</td>
      <td id="T_89860_row2_col2" class="data row2 col2" >4282.5522</td>
      <td id="T_89860_row2_col3" class="data row2 col3" >0.7982</td>
      <td id="T_89860_row2_col4" class="data row2 col4" >0.5526</td>
      <td id="T_89860_row2_col5" class="data row2 col5" >0.5890</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_89860_row3_col0" class="data row3 col0" >946.8931</td>
      <td id="T_89860_row3_col1" class="data row3 col1" >49912153.8690</td>
      <td id="T_89860_row3_col2" class="data row3 col2" >7064.8534</td>
      <td id="T_89860_row3_col3" class="data row3 col3" >0.6600</td>
      <td id="T_89860_row3_col4" class="data row3 col4" >0.5687</td>
      <td id="T_89860_row3_col5" class="data row3 col5" >0.6337</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_89860_row4_col0" class="data row4 col0" >894.9310</td>
      <td id="T_89860_row4_col1" class="data row4 col1" >39600980.8167</td>
      <td id="T_89860_row4_col2" class="data row4 col2" >6292.9310</td>
      <td id="T_89860_row4_col3" class="data row4 col3" >0.6266</td>
      <td id="T_89860_row4_col4" class="data row4 col4" >0.5442</td>
      <td id="T_89860_row4_col5" class="data row4 col5" >0.6008</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_89860_row5_col0" class="data row5 col0" >1143.7622</td>
      <td id="T_89860_row5_col1" class="data row5 col1" >29305639.8051</td>
      <td id="T_89860_row5_col2" class="data row5 col2" >5413.4684</td>
      <td id="T_89860_row5_col3" class="data row5 col3" >0.7135</td>
      <td id="T_89860_row5_col4" class="data row5 col4" >0.6892</td>
      <td id="T_89860_row5_col5" class="data row5 col5" >1.0150</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_89860_row6_col0" class="data row6 col0" >621.2060</td>
      <td id="T_89860_row6_col1" class="data row6 col1" >2229029.5746</td>
      <td id="T_89860_row6_col2" class="data row6 col2" >1492.9935</td>
      <td id="T_89860_row6_col3" class="data row6 col3" >0.9354</td>
      <td id="T_89860_row6_col4" class="data row6 col4" >0.5655</td>
      <td id="T_89860_row6_col5" class="data row6 col5" >0.6896</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_89860_row7_col0" class="data row7 col0" >988.7821</td>
      <td id="T_89860_row7_col1" class="data row7 col1" >15102101.9858</td>
      <td id="T_89860_row7_col2" class="data row7 col2" >3886.1423</td>
      <td id="T_89860_row7_col3" class="data row7 col3" >0.8676</td>
      <td id="T_89860_row7_col4" class="data row7 col4" >0.6529</td>
      <td id="T_89860_row7_col5" class="data row7 col5" >0.8995</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_89860_row8_col0" class="data row8 col0" >905.0841</td>
      <td id="T_89860_row8_col1" class="data row8 col1" >7162889.6167</td>
      <td id="T_89860_row8_col2" class="data row8 col2" >2676.3575</td>
      <td id="T_89860_row8_col3" class="data row8 col3" >0.8790</td>
      <td id="T_89860_row8_col4" class="data row8 col4" >0.5526</td>
      <td id="T_89860_row8_col5" class="data row8 col5" >0.6344</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_89860_row9_col0" class="data row9 col0" >749.6643</td>
      <td id="T_89860_row9_col1" class="data row9 col1" >7325720.3639</td>
      <td id="T_89860_row9_col2" class="data row9 col2" >2706.6068</td>
      <td id="T_89860_row9_col3" class="data row9 col3" >0.8722</td>
      <td id="T_89860_row9_col4" class="data row9 col4" >0.5652</td>
      <td id="T_89860_row9_col5" class="data row9 col5" >0.6375</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_89860_row10_col0" class="data row10 col0" >957.7835</td>
      <td id="T_89860_row10_col1" class="data row10 col1" >27505058.9964</td>
      <td id="T_89860_row10_col2" class="data row10 col2" >4657.8162</td>
      <td id="T_89860_row10_col3" class="data row10 col3" >0.7819</td>
      <td id="T_89860_row10_col4" class="data row10 col4" >0.5921</td>
      <td id="T_89860_row10_col5" class="data row10 col5" >0.7402</td>
    </tr>
    <tr>
      <th id="T_89860_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_89860_row11_col0" class="data row11 col0" >198.3695</td>
      <td id="T_89860_row11_col1" class="data row11 col1" >27686274.6063</td>
      <td id="T_89860_row11_col2" class="data row11 col2" >2410.3542</td>
      <td id="T_89860_row11_col3" class="data row11 col3" >0.1154</td>
      <td id="T_89860_row11_col4" class="data row11 col4" >0.0465</td>
      <td id="T_89860_row11_col5" class="data row11 col5" >0.1470</td>
    </tr>
  </tbody>
</table>







    Fitting 10 folds for each of 10 candidates, totalling 100 fits
    Original model was better than the tuned model, hence it will be returned. NOTE: The display metrics are for the tuned model (not the original one).
    Cell executed at: 2025-05-16 17:10:50.545619
    


```python
    # Evaluate tuned model
    print("\nEvaluating tuned model...")
    evaluate_model(tuned_model)
```

    
    Evaluating tuned model...
    


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Pipeline Plot', 'pipelin…


    Cell executed at: 2025-05-15 20:03:56.447855
    


```python
    # Save the model
    save_model(tuned_model, 'best_model')
    print("\nBest model saved as 'best_model'")
```

    Transformation Pipeline and Model Successfully Saved
    
    Best model saved as 'best_model'
    Cell executed at: 2025-05-15 20:03:56.896293
    


```python
    # Feature importance visualization
    print("\nGenerating feature importance plot...")
    try:
        if hasattr(model, 'feature_importances_'):
            # Create directory for plots if it doesn't exist
            import os
            os.makedirs('plots', exist_ok=True)
            
            plot_model(tuned_model, plot='feature', save=True, filename='plots/feature_importance')
            print("Feature importance plot saved to 'plots/feature_importance.png'")
        else:
            print("This model type doesn't support direct feature importance plotting.")
    except Exception as e:
        print(f"Could not generate feature plot: {e}")
```

    
    Generating feature importance plot...
    Could not generate feature plot: plot_model() got an unexpected keyword argument 'filename'
    Cell executed at: 2025-05-15 20:31:38.026499
    


```python
    # SHAP analysis with proper error handling
    print("\nAttempting SHAP analysis...")
    try:
        import shap
        
        # Get the preprocessed features that PyCaret actually used for training
        X_for_shap = pycaret_X
        
        # Select only numeric features for SHAP analysis if needed
        X_numeric = X_for_shap.select_dtypes(include=[np.number])
        
        # Check if any columns were dropped
        if X_for_shap.shape[1] != X_numeric.shape[1]:
            print(f"Warning: {X_for_shap.shape[1] - X_numeric.shape[1]} non-numeric columns excluded from SHAP analysis")
        
        # Create directories for output
        os.makedirs('plots', exist_ok=True)
        
        # Create SHAP explainer with additivity check disabled
        explainer = shap.Explainer(tuned_model, X_numeric)
        shap_values = explainer(X_numeric, check_additivity=False)  # Added check_additivity=False
        
        # Generate and save SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_numeric, show=False)
        plt.tight_layout()
        plt.savefig('plots/shap_summary.png')
        plt.close()
        print("SHAP analysis completed and saved as 'plots/shap_summary.png'")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Consider using a different model or approach for feature importance.")
```

    
    Attempting SHAP analysis...
    
    Available columns in X:
    ['Unnamed_0', 'Project_PRF_Year_of_Project', 'External_EEF_Industry_Sector', 'External_EEF_Organisation_Type', 'Project_PRF_Application_Type', 'Tech_TF_Primary_Programming_Language', 'Project_PRF_Functional_Size', 'Project_PRF_Relative_Size', 'Project_PRF_Normalised_Work_Effort_Level_1', 'Project_PRF_Normalised_Level_1_PDR_ufp', 'Project_PRF_Normalised_PDR_ufp', 'Project_PRF_Speed_of_Delivery', 'Project_PRF_Project_Elapsed_Time', 'Project_PRF_Team_Size_Group', 'Project_PRF_Max_Team_Size', 'Process_PMF_Development_Methodologies', 'Process_PMF_Docs', 'Tech_TF_Tools_Used', 'External_EEF_Data_Quality_Rating_B', 'Project_PRF_Application_Group_Business Application; Infrastructure Software;', 'Project_PRF_Application_Group_Infrastructure Software', 'Project_PRF_Application_Group_Mathematically intensive application', 'Project_PRF_Application_Group_Mathematically-Intensive Application', 'Project_PRF_Application_Group_Missing', 'Project_PRF_Application_Group_Real-Time Application', 'Project_PRF_Development_Type_New Development', 'Project_PRF_Development_Type_Re-development', 'Tech_TF_Development_Platform_MF', 'Tech_TF_Development_Platform_MR', 'Tech_TF_Development_Platform_Missing', 'Tech_TF_Development_Platform_Multi', 'Tech_TF_Development_Platform_PC', 'Tech_TF_Development_Platform_Proprietary', 'Tech_TF_Language_Type_3GL', 'Tech_TF_Language_Type_4GL', 'Tech_TF_Language_Type_5GL', 'Tech_TF_Language_Type_APG', 'Tech_TF_Language_Type_ApG', 'Tech_TF_Language_Type_Missing', 'Project_PRF_CASE_Tool_Used_Missing', 'Project_PRF_CASE_Tool_Used_No', 'Project_PRF_CASE_Tool_Used_Yes', 'Tech_TF_Architecture_Missing', 'Tech_TF_Architecture_Multi-tier', 'Tech_TF_Architecture_Multi-tier / Client server', 'Tech_TF_Architecture_Multi-tier with web interface', 'Tech_TF_Architecture_Multi-tier with web public interface', 'Tech_TF_Architecture_Stand alone', 'Tech_TF_Architecture_Stand-alone', 'Tech_TF_Client_Server_Missing', 'Tech_TF_Client_Server_No', 'Tech_TF_Client_Server_Not Applicable', 'Tech_TF_Client_Server_Yes', 'Tech_TF_DBMS_Used_No', 'Tech_TF_DBMS_Used_Yes']
    
    Numeric feature set created. Shape: (7058, 11)
    
    Warning: The following non-numeric columns were excluded from SHAP analysis: {'Project_PRF_Application_Group_Infrastructure Software', 'Project_PRF_Application_Group_Mathematically intensive application', 'Tech_TF_Development_Platform_PC', 'Tech_TF_Language_Type_ApG', 'Project_PRF_Application_Group_Real-Time Application', 'Tech_TF_Language_Type_3GL', 'Tech_TF_Architecture_Stand-alone', 'Project_PRF_Application_Type', 'Project_PRF_Development_Type_Re-development', 'Tech_TF_Language_Type_Missing', 'Tech_TF_Architecture_Multi-tier with web interface', 'Tech_TF_Client_Server_No', 'External_EEF_Organisation_Type', 'Project_PRF_Relative_Size', 'Project_PRF_Application_Group_Missing', 'Tech_TF_Architecture_Stand alone', 'Project_PRF_Development_Type_New Development', 'Project_PRF_Application_Group_Business Application; Infrastructure Software;', 'Project_PRF_CASE_Tool_Used_No', 'External_EEF_Industry_Sector', 'Tech_TF_Language_Type_4GL', 'Tech_TF_Architecture_Multi-tier with web public interface', 'Project_PRF_CASE_Tool_Used_Yes', 'Project_PRF_CASE_Tool_Used_Missing', 'Tech_TF_Development_Platform_Multi', 'Tech_TF_Client_Server_Not Applicable', 'Project_PRF_Team_Size_Group', 'Project_PRF_Application_Group_Mathematically-Intensive Application', 'Tech_TF_Development_Platform_Proprietary', 'Tech_TF_Architecture_Missing', 'Tech_TF_DBMS_Used_Yes', 'Tech_TF_Development_Platform_Missing', 'Tech_TF_Language_Type_5GL', 'Tech_TF_DBMS_Used_No', 'Process_PMF_Development_Methodologies', 'Tech_TF_Development_Platform_MF', 'Tech_TF_Development_Platform_MR', 'Tech_TF_Architecture_Multi-tier', 'External_EEF_Data_Quality_Rating_B', 'Tech_TF_Client_Server_Yes', 'Tech_TF_Primary_Programming_Language', 'Tech_TF_Language_Type_APG', 'Tech_TF_Client_Server_Missing', 'Tech_TF_Architecture_Multi-tier / Client server'}
    
    Creating SHAP explainer...
    SHAP explainer created successfully.
    
    Calculating SHAP values...
    

     99%|===================| 6992/7058 [00:34<00:00]        

    
    SHAP analysis failed: Additivity check failed in TreeExplainer! Please ensure the data matrix you passed to the explainer is the same shape that the model was trained on. If your data shape is correct then please report this on GitHub. This check failed because for one of the samples the sum of the SHAP values was 69485.730887, while the model output was 213759.070000. If this difference is acceptable you can set check_additivity=False to disable this check.
    Consider verifying the processed_data DataFrame and the specified target_col.
    Cell executed at: 2025-05-15 20:45:04.381878
    


```python
    # Extract feature importance directly (if available)
    print("\nExtracting direct feature importance...")
    try:
        if hasattr(tuned_model, 'feature_importances_'):
            # Use the preprocessed feature names
            fi = pd.DataFrame({
                'Feature': pycaret_X.columns,
                'Importance': tuned_model.feature_importances_
            })
            fi = fi.sort_values('Importance', ascending=False)
            print("\nFeature importances:")
            print(fi.head(15))  # Show top 15 features
            fi.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to 'feature_importance.csv'")
        else:
            print("Feature importance attribute not available for this model")
    except Exception as e:
        print(f"Failed to extract feature importance: {e}")

except Exception as e:
    print(f"\nError in PyCaret workflow: {e}")
    print("Check if PyCaret is installed correctly: pip install pycaret")

print("\nAnalysis complete!")
```

    
    Extracting direct feature importance...
    Failed to extract feature importance: name 'X' is not defined
    
    Analysis complete!
    Cell executed at: 2025-05-15 20:14:36.547142
    


```python

```


```python

```


```python

```


```python

```


```python

```
