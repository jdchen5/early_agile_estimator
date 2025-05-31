.. code:: ipython3

    # ================================
    # 1. Import libraries
    # ================================
    import pandas as pd
    import numpy as np
    import re
    from pathlib import Path
    from sklearn.preprocessing import MultiLabelBinarizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    import re
    from pathlib import Path
    import os

.. code:: ipython3

    # Sets up an automatic timestamp printout after each Jupyter cell execution 
    # and configures the default visualization style.
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


   

.. code:: ipython3

    # ================================
    # 2. Set file paths and load data
    # ================================
    data_folder = Path("../data")
    sample_file = "sample_clean_a_agile_only.xlsx"
    data_file = ""
    
    # Load the data
    print("Loading data...")
    
    file_path = f"{data_folder}/{sample_file}"  # should use data_file for model training
    file_name_no_ext = Path(file_path).stem                # 'ISBSG2016R1.1 - FormattedForCSV'
    print(file_name_no_ext)
    
    
    df = pd.read_excel(file_path)



    

.. code:: ipython3

    # functions to standardise column names
    
    def standardize_columns(df):
        return df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
    
    
    



    

.. code:: ipython3

    # ================================
    # 3. Identify columns with semicolons
    # ================================
    semicolon_cols = [
        col for col in df.columns
        if df[col].dropna().astype(str).str.contains(';').any()
    ]
    
    print("Columns with semicolons:", semicolon_cols)
    



    

.. code:: ipython3

    # ================================
    # 4. Cleaning function for semicolon-separated columns
    # ================================
    def clean_and_sort_semicolon(val, apply_standardization=False, mapping=None):
        """
        Clean, deduplicate, sort, and (optionally) standardize semicolon-separated values.
        """
        if pd.isnull(val) or val == '':
            return val
        parts = [x.strip().lower() for x in str(val).split(';') if x.strip()]
        if apply_standardization and mapping is not None:
            parts = [mapping.get(part, part) for part in parts]
        unique_cleaned = sorted(set(parts))
        return '; '.join(unique_cleaned)
    
    # Optionally: a mapping dictionary for extra standardization
    standardization_mapping = {
        "scrum": "agile development",
        "file &/or print server": "file/print server",
        # Add more business-specific mappings here!
    }




.. code:: ipython3

    # ================================
    # 5. Apply cleaning to each semicolon column
    # ================================
    for col in semicolon_cols:
        # Choose whether to apply mapping (you can edit logic below per column)
        apply_mapping = col in ['Process_PMF_Development Methodologies', 'Tech_TF_Server_Roles']
        mapping = standardization_mapping if apply_mapping else None
        df[col + "_cleaned"] = df[col].map(lambda x: clean_and_sort_semicolon(x, apply_standardization=apply_mapping, mapping=mapping))




.. code:: ipython3

    # ================================
    # 6. Show before/after for each column (first 3 examples)
    # ================================
    for col in semicolon_cols:
        print(f"\nColumn: {col}")
        print("BEFORE:", list(df[col].dropna().astype(str).unique()[:3]))
        print("AFTER:", list(df[col + "_cleaned"].dropna().astype(str).unique()[:3]))
    




.. code:: ipython3

    # ================================
    # 7. One-hot encode cleaned columns & show unique categories
    # ================================
    unique_values = {}
    mlb_results = {}
    
    for col in semicolon_cols:
        cleaned_col = col + "_cleaned"
        values = df[cleaned_col].dropna().astype(str).apply(lambda x: [item.strip() for item in x.split(';') if item.strip()])
        mlb = MultiLabelBinarizer()
        onehot = pd.DataFrame(
            mlb.fit_transform(values),
            columns=[f"{cleaned_col}__{cat}" for cat in mlb.classes_],
            index=values.index
        )
        # Merge one-hot with main df if needed: df = df.join(onehot)
        mlb_results[cleaned_col] = onehot
        unique_values[col] = list(mlb.classes_)
        print(f"\nUnique categories in '{col}':\n", mlb.classes_)
    




.. code:: ipython3

    # ================================
    # 8. (Optional) Export cleaned data & one-hot encoded columns
    # ================================
    df.to_csv(data_folder / (file_name_no_ext + "_cleaned_data.csv"), index=False)
    
    # For one-hot: 
    pd.concat([df, onehot], axis=1).to_csv(data_folder / (file_name_no_ext + "_cleaned_data_with_onehot.csv"), index=False)



.. code:: ipython3

    # Step 1: Replace original columns with cleaned versions
    for col in semicolon_cols:
        cleaned_col = col + "_cleaned"
        if cleaned_col in df.columns:
            df[col] = df[cleaned_col]
    
    # Step 2: Drop the now-redundant _cleaned columns
    df = df.drop([col + "_cleaned" for col in semicolon_cols if col + "_cleaned" in df.columns], axis=1)
    
    df_cleaned = standardize_columns(df)
    
    # Step 3: Save the cleaned DataFrame to CSV
    df_cleaned.to_csv(data_folder / (file_name_no_ext + "_cleaned_no_add.csv"), index=False)



    

.. code:: ipython3

    print("Current columns:", df_cleaned.columns.tolist())






