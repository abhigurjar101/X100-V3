import pandas as pd
import numpy as np
import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase, alphanumeric with underscores."""
    df_clean = df.copy()
    new_cols = []
    for col in df_clean.columns:
        # Replace special characters and spaces with underscores
        col_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(col))
        col_clean = re.sub(r'_+', '_', col_clean).strip('_').lower()
        new_cols.append(col_clean)
    df_clean.columns = new_cols
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes exact duplicate rows and returns cleaned df and count of removed rows."""
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    removed_count = initial_count - len(df_clean)
    return df_clean, removed_count

def impute_missing_values(df: pd.DataFrame, strategy: str = "auto") -> tuple[pd.DataFrame, dict]:
    """
    Imputes missing values in the dataframe.
    Strategies:
      - 'auto': numeric with median, categorical with mode or 'Unknown'
      - 'drop': drop rows with missing values
      - 'mean': numeric with mean
      - 'median': numeric with median
      - 'mode': numeric and categorical with mode
    Returns cleaned df and a summary dict of actions taken.
    """
    df_clean = df.copy()
    actions = {}
    
    if strategy == "drop":
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        actions["dropped_rows"] = initial_rows - len(df_clean)
        return df_clean, actions
        
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count == 0:
            continue
            
        col_type = df_clean[col].dtype
        
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            if strategy in ["auto", "median"]:
                val = df_clean[col].median()
                method = "median"
            elif strategy == "mean":
                val = df_clean[col].mean()
                method = "mean"
            else:
                val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                method = "mode"
                
            df_clean[col] = df_clean[col].fillna(val)
            actions[col] = f"Imputed {null_count} nulls with {method} ({val:.2f})"
            
        else:
            # Categorical or text
            if not df_clean[col].mode().empty:
                val = df_clean[col].mode()[0]
                method = "mode"
            else:
                val = "Unknown"
                method = "'Unknown'"
                
            df_clean[col] = df_clean[col].fillna(val)
            actions[col] = f"Imputed {null_count} nulls with {method} ('{val}')"
            
    return df_clean, actions

def auto_convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attempts to auto-convert string date columns to datetime and numeric strings to floats/ints."""
    df_clean = df.copy()
    conversions = {}
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try datetime
            try:
                # Check if it looks like a date format string
                sample_vals = df_clean[col].dropna().head(10).astype(str)
                if sample_vals.str.contains(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', regex=True).all():
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    conversions[col] = "Converted object to datetime64"
                    continue
            except Exception:
                pass
                
            # Try numeric
            try:
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                # If conversion succeeded without introducing too many new NaNs
                if numeric_series.notnull().sum() >= df_clean[col].notnull().sum() * 0.9:
                    df_clean[col] = numeric_series
                    conversions[col] = "Converted object to numeric (float/int)"
            except Exception:
                pass
                
    return df_clean, conversions

def run_full_pipeline(df: pd.DataFrame, impute_strategy: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Runs standard cleaning pipeline and returns report."""
    report = {}
    
    # 1. Standardize columns
    df_step1 = clean_column_names(df)
    report["column_names_standardized"] = True
    
    # 2. Convert types
    df_step2, type_conversions = auto_convert_types(df_step1)
    report["type_conversions"] = type_conversions
    
    # 3. Remove duplicates
    df_step3, removed_dups = remove_duplicates(df_step2)
    report["removed_duplicates_count"] = removed_dups
    
    # 4. Impute missing
    df_final, impute_actions = impute_missing_values(df_step3, strategy=impute_strategy)
    report["imputations"] = impute_actions
    
    return df_final, report
