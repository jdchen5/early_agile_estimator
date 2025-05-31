PyCaret: ISBSG Data Analysis & Regression

.. code:: ipython3

    # <span style="color: blue;">ISBSG Data Analysis & Regression</span>
    

.. code:: ipython3

    import sys
    
    print(sys.executable)


   

.. code:: ipython3

    # # ISBSG Data Analysis and Regression Modeling
    # 
    # This notebook performs data cleaning, preprocessing, and regression modeling on the ISBSG dataset.
    
    # ## Setup and Environment Configuration
    
    # Install required packages (uncomment if needed)
    #!pip install -r "../requirements.txt" --only-binary=all

.. code:: ipython3

    # Import basic libraries
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pycaret
    from datetime import datetime
    import re
    import seaborn as sns
    import sklearn
    import shap

.. code:: ipython3

    # Define the foler path
    models_folder = '../models'
    skeleton_models_folder = '../skeleton_models'
    plots_folder = '../plots'
    temp_folder = '../temp'
    data_folder = '../data'
    logs_folder = '../logs'
    sample_file = 'sample_clean_a_agile_only_cleaned_no_add.csv'
    data_file = 'ISBSG2016R1_1_Formatted4CSVAgileOnly_cleaned'
    
    # Identify target column
    TARGET_COL = 'Project_PRF_Normalised_Work_Effort'
    print(f"\nTarget variable: '{TARGET_COL}'")

Table of Content
================

In this notebook you will apply xxxxxxx

- `Part 1 <#part1>`__- Data Loading and Initial Exploration
- `Part 2 <#part2>`__- Data Cleaning and Preprocessing
- `Part 3 <#part3>`__- Data Profiling
- `Part 4 <#part4>`__- Module Building with PyCaret
- `Part 5 <#part5>`__- Model Preparation
- `Part 6 <#part6>`__- Baseline Modeling and Evaluation
- `Part 7 <#part7>`__- Advanced Modeling and Hyperparameter Tuning
- `Part 8 <#part8>`__- Model Comparison and Selection
- `Part 9 <#part9>`__- End

.. code:: ipython3

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
    
    # Setup timestamp callback
    setup_timestamp_callback()
    
    # Set visualization style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)





`Back to top <#Index:>`__

Part 1 -Data Loading and Initial Exploration
============================================

This section is dedicated to loading the dataset, performing initial
data exploration such as viewing the first few rows, and summarizing the
dataset’s characteristics, including missing values and basic
statistical measures.

.. code:: ipython3

    # Load the data
    
    from pathlib import Path
    
    print("Loading data...")
    
    file_path = f"{data_folder}/{sample_file}"  #should use data_file
    file_name_no_ext = Path(file_path).stem                # 'ISBSG2016R1.1 - FormattedForCSV'
    print(file_name_no_ext)
    
    
    df = pd.read_csv(file_path)
    




.. code:: ipython3

    def display_header(text):
        try:
            from IPython.display import display, Markdown
            display(Markdown(f"# {text}"))
        except ImportError:
            print(f"\n=== {text} ===\n")
    
    def display_subheader(text):
        try:
            from IPython.display import display, Markdown
            display(Markdown(f"## {text}"))
        except ImportError:
            print(f"\n-- {text} --\n")
    
    def explore_data(df: pd.DataFrame) -> None:
        """
        Perform exploratory data analysis on the input DataFrame with nicely aligned plots.
        Args:
            df: Input DataFrame
        """
        from IPython.display import display
    
        display_header("Exploratory Data Analysis")
        
        # Data Overview
        display_subheader("Data Overview")
        print(f"Dataset shape: {df.shape}")
        if df.shape[0] > 20:
            print("First 5 rows:")
            display(df.head())
            print("Last 5 rows:")
            display(df.tail())
        else:
            display(df)
        
        # Duplicate Row Checking
        display_subheader("Duplicate Rows")
        num_duplicates = df.duplicated().sum()
        print(f"Number of duplicate rows: {num_duplicates}")
    
        # Data Types and Memory Usage
        display_subheader("Data Types and Memory Usage")
        dtype_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Memory Usage (MB)': df.memory_usage(deep=True) / 1024 / 1024
        })
        display(dtype_info)
        
        # Unique Values Per Column
        display_subheader("Unique Values Per Column")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
        
        # Type Conversion Suggestions
        display_subheader("Type Conversion Suggestions")
        potential_cat = [
            col for col in df.select_dtypes(include=['object']).columns
            if df[col].nunique() < max(30, 0.05*df.shape[0])
        ]
        if potential_cat:
            print("Consider converting to 'category' dtype for memory/performance:")
            print(potential_cat)
        else:
            print("No obvious candidates for 'category' dtype conversion.")
        
        # Summary Statistics
        display_subheader("Summary Statistics")
        try:
            display(df.describe(include='all').T.style.background_gradient(cmap='Blues', axis=1))
        except Exception:
            display(df.describe(include='all').T)
        
        # Missing Values
        display_subheader("Missing Values")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing,
            'Percentage (%)': missing_percent.round(2)
        })
        if missing.sum() > 0:
            display(missing_info[missing_info['Missing Values'] > 0]
                    .sort_values('Missing Values', ascending=False)
                    .style.background_gradient(cmap='Reds'))
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            cols_with_missing = missing_info[missing_info['Missing Values'] > 0].index
            if len(cols_with_missing) > 0:
                sns.heatmap(df[cols_with_missing].isnull(), 
                            cmap='viridis', 
                            yticklabels=False, 
                            cbar_kws={'label': 'Missing Values'})
                plt.title('Missing Value Patterns')
                plt.tight_layout()
                plt.show()
        else:
            print("No missing values in the dataset.")
        
        # Numerical Distributions
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numerical_cols) > 0:
            display_subheader("Distribution of Numerical Features")
            sample_cols = numerical_cols[:min(12, len(numerical_cols))]
            num_cols = len(sample_cols)
            num_rows = (num_cols + 2) // 3  # 3 plots per row, rounded up
            fig = plt.figure(figsize=(18, num_rows * 4))
            grid = plt.GridSpec(num_rows, 3, figure=fig, hspace=0.4, wspace=0.3)
            for i, col in enumerate(sample_cols):
                row, col_pos = divmod(i, 3)
                ax = fig.add_subplot(grid[row, col_pos])
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue', alpha=0.7)
                mean_val = df[col].mean()
                median_val = df[col].median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')
                stats_text = (f"Std: {df[col].std():.2f}\n"
                              f"Min: {df[col].min():.2f}\n"
                              f"Max: {df[col].max():.2f}")
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
                ax.set_title(f'Distribution of {col}')
                ax.legend(fontsize='small')
            plt.tight_layout()
            plt.show()
            # Correlation matrix and top correlations
            if len(numerical_cols) > 1:
                display_subheader("Correlation Matrix")
                corr = df[numerical_cols].corr().round(2)
                mask = np.triu(np.ones_like(corr, dtype=bool))
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                            fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, 
                            annot_kws={"size": 10})
                plt.title('Correlation Matrix (Lower Triangle Only)', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                plt.show()
                # Top correlations
                if len(numerical_cols) > 5:
                    corr_unstack = corr.unstack()
                    corr_abs = corr_unstack.apply(abs)
                    corr_abs = corr_abs[corr_abs < 1.0]
                    highest_corrs = corr_abs.sort_values(ascending=False).head(15)
                    display_subheader("Top Correlations")
                    for (col1, col2), corr_val in highest_corrs.items():
                        actual_val = corr.loc[col1, col2]
                        print(f"{col1} — {col2}: {actual_val:.2f}")
                    pairs_to_plot = [(idx[0], idx[1]) for idx in highest_corrs.index][:6]
                    if pairs_to_plot:
                        fig = plt.figure(figsize=(18, 12))
                        grid = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
                        for i, (col1, col2) in enumerate(pairs_to_plot):
                            row, col_pos = divmod(i, 3)
                            ax = fig.add_subplot(grid[row, col_pos])
                            sns.regplot(x=df[col1], y=df[col2], ax=ax, scatter_kws={'alpha':0.5})
                            r_value = df[col1].corr(df[col2])
                            ax.set_title(f'{col1} vs {col2} (r = {r_value:.2f})')
                        plt.tight_layout()
                        plt.show()
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            display_subheader("Categorical Features")
            sample_cat_cols = categorical_cols[:min(6, len(categorical_cols))]
            num_cat_cols = len(sample_cat_cols)
            num_cat_rows = (num_cat_cols + 1) // 2
            fig = plt.figure(figsize=(18, num_cat_rows * 5))
            grid = plt.GridSpec(num_cat_rows, 2, figure=fig, hspace=0.4, wspace=0.2)
            for i, col in enumerate(sample_cat_cols):
                row, col_pos = divmod(i, 2)
                ax = fig.add_subplot(grid[row, col_pos])
                value_counts = df[col].value_counts().sort_values(ascending=False)
                top_n = min(10, len(value_counts))
                if len(value_counts) > top_n:
                    top_values = value_counts.head(top_n-1)
                    other_count = value_counts.iloc[top_n-1:].sum()
                    plot_data = pd.concat([top_values, pd.Series({'Other': other_count})])
                else:
                    plot_data = value_counts
                sns.barplot(x=plot_data.values, y=plot_data.index, ax=ax, palette='viridis')
                ax.set_title(f'Distribution of {col} (Total: {len(value_counts)} unique values)')
                ax.set_xlabel('Count')
                total = plot_data.sum()
                for j, v in enumerate(plot_data.values):
                    percentage = v / total * 100
                    ax.text(v + 0.1, j, f'{percentage:.1f}%', va='center')
            plt.tight_layout()
            plt.show()
            # Categorical-numerical boxplots
            if numerical_cols and len(categorical_cols) > 0:
                display_subheader("Categorical-Numerical Relationships")
                numerical_variances = df[numerical_cols].var()
                target_numerical = numerical_variances.idxmax()
                sample_cat_for_box = [col for col in categorical_cols 
                                      if df[col].nunique() <= 15][:4]
                if sample_cat_for_box:
                    fig = plt.figure(figsize=(18, 5 * len(sample_cat_for_box)))
                    for i, cat_col in enumerate(sample_cat_for_box):
                        ax = fig.add_subplot(len(sample_cat_for_box), 1, i+1)
                        order = df.groupby(cat_col)[target_numerical].median().sort_values().index
                        sns.boxplot(x=cat_col, y=target_numerical, data=df, ax=ax, 
                                    order=order, palette='Set3')
                        ax.set_title(f'{cat_col} vs {target_numerical}')
                        ax.set_xlabel(cat_col)
                        ax.set_ylabel(target_numerical)
                        plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
    
    # Exploratory Data Analysis
    explore_data(df)
    



Exploratory Data Analysis
=========================



Data Overview
-------------


   

.. code:: ipython3

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




.. code:: ipython3

    # Create a mapping from original to cleaned column names
    column_mapping = dict(zip(original_columns, df.columns))
    print("\nColumn name mapping (original -> cleaned):")
    for orig, clean in column_mapping.items():
        if orig != clean:  # Only show columns that changed
            print(f"  '{orig}' -> '{clean}'")
    


    

.. code:: ipython3

    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    



    

.. code:: ipython3

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
    


    

`Back to top <#Index:>`__

Part 2 - Data Cleaning and Preprocessing
========================================

Here, data cleaning tasks like handling missing values and providing a
detailed summary of each feature, including its type, number of unique
values, and a preview of unique values, are performed.

.. code:: ipython3

    # Analyse missing values
    print("\nAnalysing missing values...")
    missing_pct = df.isnull().mean() * 100
    missing_sorted = missing_pct.sort_values(ascending=False)
    print("Top 10 columns with highest missing percentages:")
    print(missing_sorted)


.. parsed-literal::

    
    Analysing missing values...
    Top 10 columns with highest missing percentages:
    People_PRF_Project_user_involvement              100.000000
    People_PRF_IT_experience_less_than_1_yr          100.000000
    Tech_TF_ClientServer_Description                 100.000000
    Tech_TF_Type_of_Server                           100.000000
    People_PRF_IT_experience_1_to_3_yr               100.000000
    People_PRF_IT_experience_great_than_3_yr         100.000000
    Process_PMF_Prototyping_Used                      93.589744
    People_PRF_BA_team_experience_less_than_1_yr      80.769231
    Project_PRF_CASE_Tool_Used                        79.487179
    People_PRF_BA_team_experience_great_than_3_yr     78.205128
    People_PRF_IT_experience_great_than_9_yr          78.205128
    Project_PRF_Currency_multiple                     76.923077
    People_PRF_BA_team_experience_1_to_3_yr           76.923077
    People_PRF_IT_experience_3_to_9_yr                75.641026
    People_PRF_Project_manage_experience              75.641026
    People_PRF_IT_experience_less_than_3_yr           74.358974
    Tech_TF_Client_Roles                              73.076923
    Tech_TF_Server_Roles                              71.794872
    Project_PRF_Total_project_cost                    69.230769
    Project_PRF_Cost_currency                         67.948718
    People_PRF_Personnel_changes                      65.384615
    Tech_TF_Client_Server                             65.384615
    People_PRF_Project_manage_changes                 65.384615
    Project_PRF_Defect_Density                        65.384615
    Tech_TF_Web_Development                           64.102564
    Project_PRF_Manpower_Delivery_Rate                38.461538
    Tech_TF_DBMS_Used                                 37.179487
    Project_PRF_Max_Team_Size                         34.615385
    Project_PRF_Team_Size_Group                       34.615385
    Tech_TF_Development_Platform                      19.230769
    Tech_TF_Architecture                              19.230769
    Project_PRF_Application_Group                      6.410256
    Project_PRF_Speed_of_Delivery                      3.846154
    Project_PRF_Project_Elapsed_Time                   2.564103
    Project_PRF_Functional_Size                        1.282051
    Project_PRF_Relative_Size                          1.282051
    Project_PRF_Normalised_PDR_ufp                     1.282051
    External_EEF_Industry_Sector                       1.282051
    Project_PRF_Normalised_Level_1_PDR_ufp             1.282051
    ISBSG_Project_ID                                   0.000000
    Tech_TF_Tools_Used                                 0.000000
    External_EEF_Data_Quality_Rating                   0.000000
    Process_PMF_Development_Methodologies              0.000000
    Project_PRF_Normalised_Work_Effort                 0.000000
    Project_PRF_Normalised_Work_Effort_Level_1         0.000000
    Tech_TF_Primary_Programming_Language               0.000000
    Tech_TF_Language_Type                              0.000000
    Project_PRF_Development_Type                       0.000000
    Project_PRF_Application_Type                       0.000000
    External_EEF_Organisation_Type                     0.000000
    Project_PRF_Year_of_Project                        0.000000
    Process_PMF_Docs                                   0.000000
    dtype: float64
    Cell executed at: 2025-05-29 20:49:57.705622
    

.. code:: ipython3

    # Identify columns with high missing values (>70%)
    high_missing_cols = missing_pct[missing_pct > 70].index.tolist()
    print(f"\nColumns with >70% missing values ({len(high_missing_cols)} columns):")
    for col in high_missing_cols[:]:  # Show first 5
        print(f"  - {col}: {missing_pct[col]:.2f}% missing")
    if len(high_missing_cols) > 5:
        print(f"  - ... and {len(high_missing_cols) - 5} more columns")



    

.. code:: ipython3

    # Create a clean dataframe by dropping high-missing columns
    
    cols_to_keep = ['Project_PRF_CASE_Tool_Used', 'Process_PMF_Prototyping_Used', 'Tech_TF_Client_Roles', 'Tech_TF_Type_of_Server', 'Tech_TF_ClientServer_Description']
    
    # Filter high_missing_cols to remove any you want to keep
    final_high_missing_cols = [col for col in high_missing_cols if col not in cols_to_keep]
    
    
    df_clean = df.drop(columns=final_high_missing_cols)
    print(f"\nData shape after dropping high-missing columns: {df_clean.shape}")
    print(f"\nHigh missing columns got dropped are: {final_high_missing_cols}")
    
    # Numerical columns
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    print("\nNumerical columns:")
    print(num_cols)
    
    # Categorical columns (object or category dtype)
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\nCategorical columns:")
    print(cat_cols)
    
    



.. code:: ipython3

    # Handle remaining missing values
    print("\nHandling remaining missing values...")



    

.. code:: ipython3

    # Fill missing values in categorical columns with "Missing"
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_clean[col].fillna('Missing', inplace=True)




.. code:: ipython3

    # Check remaining missing values
    remaining_missing = df_clean.isnull().sum()
    remaining_missing_count = sum(remaining_missing > 0)
    print(f"\nColumns with remaining missing values: {remaining_missing_count}")
    if remaining_missing_count > 0:
        print("Top columns with missing values:")
        print(remaining_missing[remaining_missing > 0].sort_values(ascending=False))



    

.. code:: ipython3

    print(df_clean.columns.tolist())
    



.. code:: ipython3

    # Verify target variable
    print(f"\nTarget variable '{TARGET_COL}' summary:")
    print(f"Unique values: {df_clean[TARGET_COL].nunique()}")
    print(f"Missing values: {df_clean[TARGET_COL].isnull().sum()}")
    print(f"Top value counts:")
    print(df_clean[TARGET_COL].value_counts().head())
    



.. code:: ipython3

    # Check for infinite values
    inf_check = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nNumber of infinite values: {inf_check}")


    

.. code:: ipython3

    # Save cleaned data
    
    file_name_no_ext
    
    df_clean.to_csv(f"{data_folder}/{file_name_no_ext}_dropped.csv", index=False)
    print(f'{data_folder}/{file_name_no_ext}_dropped.csv')
    


   

`Back to top <#Index:>`__

Part 3 - Feature Engineering and Selection
==========================================

Involves creating or selecting specific features for the model based on
insights from EDA, including handling categorical variables and reducing
dimensionality if necessary.

.. code:: ipython3

    # Identify categorical columns and check cardinality
    print("\nCategorical columns and their cardinality:")
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:5]:  # Show first 5
        print(f"  {col}: {df_clean[col].nunique()} unique values")
    if len(cat_cols) > 5:
        print(f"  ... and {len(cat_cols) - 5} more columns")



    

.. code:: ipython3

    # ================================
    # Identify columns with semicolons
    # ================================
    semicolon_cols = [
        col for col in df_clean.columns
        if df_clean[col].dropna().astype(str).str.contains(';').any()
    ]
    
    print("Columns with semicolons:", semicolon_cols)
    



.. code:: ipython3

    # One-hot encode categorical columns with low cardinality (<10 unique values)
    low_card_cols = [col for col in cat_cols if df_clean[col].nunique() < 10]
    print(f"\nWill apply one-hot encoding to {len(low_card_cols)} low-cardinality columns:")
    for col in low_card_cols[:]:  # Show first 5
        print(f"  - {col}")
    if len(low_card_cols) > 5:
        print(f"  - ... and {len(low_card_cols) - 5} more columns")
    


    

.. code:: ipython3

    # Create encoded dataframe for single-value columns
    
    multi_value_low_card_cols = list(set(low_card_cols) & set(semicolon_cols))
    
    # Filter low_card_cols to remove any multi_value_cols
    final_low_card_cols = [col for col in low_card_cols if col not in semicolon_cols]
    
    
    df_single = pd.get_dummies(df_clean, columns=final_low_card_cols, drop_first=True)
    
    encoded_columns = df_encoded.columns.tolist()
    print(f"\nData shape after one-hot encoding: {df_encoded.shape}")
    print("\nAll column names:")
    print(df_encoded.columns.tolist())
    



.. code:: ipython3

    # One-hot encode multi-value columns
    from sklearn.preprocessing import MultiLabelBinarizer
    
    for col in multi_value_low_card_cols:
        values = df_clean[col].dropna().astype(str).apply(lambda x: [v.strip() for v in x.split(';') if v.strip()])
        mlb = MultiLabelBinarizer()
        onehot = pd.DataFrame(
            mlb.fit_transform(values),
            columns=[f"{col}__{cat}" for cat in mlb.classes_],
            index=values.index
        )
        # Merge with df_single by index (aligns correctly)
        df_single = df_single.join(onehot, how='left')
    
    df_encoded = df_single
    
    df_encoded = df_encoded.drop(columns=multi_value_low_card_cols)


   

.. code:: ipython3

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
    
    encoded_columns_fixed = df_fixed.columns.tolist()
    
    # Print some example fixed columns to verify
    print("\nSample of fixed column names:")
    for i, (old, new) in enumerate(zip(df_encoded.columns[:15], df_fixed.columns[:15])):
        print(f"Original: {old} -> Fixed: {new}")



    

.. code:: ipython3

    # Save this DataFrame with fixed column names
    
    df_fixed.to_csv(f'{data_folder}/{file_name_no_ext}_fixed_columns_data.csv', index=False)
    print(f"Saved data with fixed column names to '{data_folder}/{file_name_no_ext}_fixed_columns_data.csv'")


   

.. code:: ipython3

    # Create a diagnostic file with all column transformations
    with open(f'{temp_folder}/{file_name_no_ext}_column_transformations.txt', 'w') as f:
        f.write("Column name transformations:\n")
        f.write("--------------------------\n")
        for old, new in zip(df_encoded.columns, df_fixed.columns):
            f.write(f"{old} -> {new}\n")
    print(f"Saved complete column transformations to '{temp_folder}/{file_name_no_ext}_column_transformations.txt'")


    

`Back to top <#Index:>`__

Part 4 - Data Profiling
=======================

xxx

.. code:: ipython3

    # ## Data Profiling (Optional)
    
    try:
        from ydata_profiling import ProfileReport
        
        print("\nGenerating data profile report...")
        profile = ProfileReport(df_clean, title="ISBSG Dataset Profiling Report", minimal=True)
        profile.to_file(f"{data_folder}/{file_name_no_ext}_data_profile.html")
        print(f"Data profile report saved to '{data_folder}/{file_name_no_ext}_data_profile.html'")
    except ImportError:
        print("\nSkipping data profiling (ydata_profiling not installed)")
        print("To install: pip install ydata-profiling")



.. raw:: html

    
    <div>
        <ins><a href="https://ydata.ai/register">Upgrade to ydata-sdk</a></ins>
        <p>
            Improve your data and profiling with ydata-sdk, featuring data quality scoring, redundancy detection, outlier identification, text validation, and synthetic data generation.
        </p>
    </div>
    




`Back to top <#Index:>`__

Part 5 - PyCaret setup
======================

xxx

.. code:: ipython3

    print(sklearn.__version__)
    print(pycaret.__version__)  


    

.. code:: ipython3

    from pycaret.regression import setup, get_config
    from sklearn.preprocessing import StandardScaler
    import os
    
    ignore_cols = ['isbsg_project_id', 'external_eef_data_quality_rating', 'external_eef_data_quality_rating_b', 'project_prf_normalised_work_effort_level_1', 'project_prf_normalised_level_1_pdr_ufp', 'project_prf_normalised_pdr_ufp', 
                   'project_prf_project_elapsed_time', 'people_prf_ba_team_experience_less_than_1_yr', 'people_prf_ba_team_experience_1_to_3_yr', 
                   'people_prf_ba_team_experience_great_than_3_yr', 'people_prf_it_experience_less_than_1_yr', 'people_prf_it_experience_1_to_3_yr', 
                   'people_prf_it_experience_great_than_3_yr', 'people_prf_it_experience_less_than_3_yr', 'people_prf_it_experience_3_to_9_yr', 
                   'people_prf_it_experience_great_than_9_yr', 'people_prf_project_manage_experience', 'project_prf_total_project_cost', 
                   'project_prf_cost_currency', 'project_prf_currency_multiple', 'project_prf_speed_of_delivery', 'people_prf_project_manage_changes', 
                   'project_prf_defect_density','project_prf_manpower_delivery_rate'
                ]
    
    print(f"Final encoded feature list: {encoded_columns_fixed}")
    print(f"\nIgnred feature columns: {ignore_cols}")
    setup_results = setup(
        data=df_fixed,
        target=TARGET_COL,
        ignore_features=ignore_cols,
        session_id=123,
        preprocess=True,
        # Add these lines to enable normalization (scaling)
        normalize=True,             # This will use StandardScaler (Z-score normalization) by default
        normalize_method='zscore',  # Explicitly state 'zscore', or choose 'minmax', 'maxabs', 'robust'
        verbose=False
    )
    
    # Get the fitted pipeline from PyCaret
    preprocessor = get_config('pipeline')
    
    # --- Capture the scaler model ---
    # Access the 'normalize' step from the pipeline's named_steps
    # The actual scaler object is inside the 'transformer' attribute of the TransformerWrapper
    scaler_model = preprocessor.named_steps['normalize'].transformer
    
    # Create the models folder if it doesn't exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"Created folder: {models_folder}")
    
    # Define the file path for the scaler model
    scaler_filename = os.path.join(models_folder, 'standard_scaler.pkl') # .pkl is a common extension for pickled files
    
    # Create the models folder if it doesn't exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"Created folder: {models_folder}")
    
    # Save the scaler model
    joblib.dump(scaler_model, scaler_filename)
    print(f"Scaler model saved to: {scaler_filename}")
    
    # You can now print it to verify
    print(scaler_model)
    
    # You can also check its type
    print(type(scaler_model))
    
    # If it's a StandardScaler, it will have .mean_ and .scale_ attributes after fitting
    if isinstance(scaler_model, StandardScaler):
        print(f"Scaler Mean: {scaler_model.mean_}")
        print(f"Scaler Scale (Std Dev): {scaler_model.scale_}")
    
    # --- Example of using the captured scaler (on new data, assuming it's in the same format) ---
    # Note: You typically use the entire PyCaret pipeline for new data,
    # but if you specifically needed just the scaler for some custom preprocessing,
    # you could do it like this:
    #
    # # Assuming 'new_numerical_data' is a pandas DataFrame or numpy array
    # # containing only the numerical features that were scaled by PyCaret
    # # (i.e., 'project_prf_year_of_project', 'project_prf_functional_size', etc.)
    # scaled_data_custom = scaler_model.transform(new_numerical_data)
    # print(scaled_data_custom)
    
    
    



`Back to top <#Index:>`__

Part 6 - Feature Correlation Analysis
=====================================

xxx

.. code:: ipython3

    # Feature correlation analysis
    print("\nAnalyzing feature correlations...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from pycaret.regression import get_config
    
        # Create directory for plots
        os.makedirs(plots_folder, exist_ok=True)
    
        # Get data from PyCaret
        X = get_config('X')
    
        # Ensure we're working with numeric data only
        X_numeric = X.select_dtypes(include=[np.number])
    
        # Drop rows with NaN or Inf values before correlation and VIF analysis
        X_numeric_clean = X_numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
        # Get number of features
        n_features = X_numeric_clean.shape[1]
        print(f"Analysing correlations among {n_features} numeric features")
    
        # Calculate correlation matrix
        corr_matrix = X_numeric_clean.corr()
    
        # Determine features with high correlation
        correlation_threshold = 0.7
        high_corr_pairs = []
    
        # Find highly correlated feature pairs
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    high_corr_pairs.append((
                        X_numeric_clean.columns[i],
                        X_numeric_clean.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
    
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(corr_matrix)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
        # If there are too many features, show only the ones with high correlation
        if n_features > 20:
            print(f"Large number of features detected ({n_features}). Creating filtered correlation matrix.")
            # Get list of features with high correlation
            high_corr_features = set()
            for feat1, feat2, _ in high_corr_pairs:
                high_corr_features.add(feat1)
                high_corr_features.add(feat2)
    
            # If there are high correlations, show only those features
            if high_corr_features:
                high_corr_features = list(high_corr_features)
                filtered_corr = corr_matrix.loc[high_corr_features, high_corr_features]
    
                # Plot filtered heatmap
                sns.heatmap(filtered_corr, mask=np.triu(filtered_corr),
                            cmap=cmap, vmax=1, vmin=-1, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5},
                            annot=True, fmt=".2f")
                plt.title('Correlation Heatmap (Filtered to Highly Correlated Features)')
            else:
                # No high correlations, show full matrix
                sns.heatmap(corr_matrix, mask=mask,
                            cmap=cmap, vmax=1, vmin=-1, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5})
                plt.title('Correlation Heatmap (All Features)')
        else:
            # For smaller feature sets, show the full correlation matrix
            sns.heatmap(corr_matrix, mask=mask,
                        cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5},
                        annot=True, fmt=".2f")
            plt.title('Correlation Heatmap (All Features)')
    
        plt.tight_layout()
        plt.savefig(f"{plots_folder}/{file_name_no_ext}_correlation_heatmap.png")
        plt.show()      # <-- Show in notebook
        plt.close()
        print("Correlation heatmap saved as {plots_folder}/{file_name_no_ext}_correlation_heatmap.png")
    
        # Calculate Variance Inflation Factor (VIF) if there are enough samples
        vif_data = None
        if X_numeric_clean.shape[0] > X_numeric_clean.shape[1]:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
    
                # Calculate VIF for each feature
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X_numeric_clean.columns
                vif_data["VIF"] = [variance_inflation_factor(X_numeric_clean.values, i)
                                   for i in range(X_numeric_clean.shape[1])]
    
                # Sort by VIF value
                vif_data = vif_data.sort_values("VIF", ascending=False)
    
                # Plot VIF values
                plt.figure(figsize=(12, 8))
                plt.barh(vif_data["Feature"], vif_data["VIF"])
                plt.axvline(x=5, color='r', linestyle='--', label='VIF=5 (Moderate multicollinearity)')
                plt.axvline(x=10, color='darkred', linestyle='--', label='VIF=10 (High multicollinearity)')
                plt.xlabel('VIF Value')
                plt.title('Variance Inflation Factor (VIF) for Features')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{plots_folder}/{file_name_no_ext}_vif_values.png")
                plt.show()      # <-- Show in notebook
                plt.close()
                print(f"VIF values plot saved as {plots_folder}/{file_name_no_ext}_vif_values.png")
            except Exception as vif_err:
                print(f"Could not calculate VIF: {vif_err}")
        else:
            print("Not enough samples to calculate VIF (need more samples than features)")
    
        # Print results
        print(f"\nFound {len(high_corr_pairs)} feature pairs with correlation > {correlation_threshold}:")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  • {feat1} and {feat2}: {corr:.4f}")
    
        # Print VIF results if available
        if vif_data is not None:
            high_vif_threshold = 10
            high_vif_features = vif_data[vif_data["VIF"] > high_vif_threshold]
            if not high_vif_features.empty:
                print(f"\nFeatures with high VIF (> {high_vif_threshold}):")
                for _, row in high_vif_features.iterrows():
                    print(f"  • {row['Feature']}: {row['VIF']:.2f}")
            else:
                print(f"\nNo features have VIF > {high_vif_threshold}")
    
        # Recommendations based on analysis
        print("\n--- Multicollinearity Analysis Recommendations ---")
        if high_corr_pairs:
            print("Consider addressing multicollinearity by:")
            print("1. Removing one feature from each highly correlated pair")
            print("2. Creating new features by combining correlated features")
            print("3. Applying dimensionality reduction techniques like PCA")
    
            # Identify top candidates for removal
            if len(high_corr_pairs) > 0:
                print("\nPotential candidates for removal:")
                # Count frequency of each feature in high correlation pairs
                freq = {}
                for feat1, feat2, _ in high_corr_pairs:
                    freq[feat1] = freq.get(feat1, 0) + 1
                    freq[feat2] = freq.get(feat2, 0) + 1
    
                # Features that appear most frequently in high correlation pairs
                freq_df = pd.DataFrame({'Feature': list(freq.keys()),
                                        'Frequency in high corr pairs': list(freq.values())})
                freq_df = freq_df.sort_values('Frequency in high corr pairs', ascending=False)
    
                for _, row in freq_df.head(5).iterrows():
                    print(f"  • {row['Feature']} (appears in {row['Frequency in high corr pairs']} high correlation pairs)")
        else:
            print("No significant multicollinearity detected based on correlation analysis.")
    
        if vif_data is not None and not high_vif_features.empty:
            print("\nBased on VIF analysis, consider removing or transforming these features with high VIF values.")
    
    except Exception as e:
        print(f"Feature correlation analysis failed: {e}")
    




