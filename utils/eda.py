import pandas as pd
import numpy as np

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Returns general overview metrics of the dataframe."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": len(df.select_dtypes(include=['datetime', 'datetime64', '<M8[ns]']).columns),
        "total_missing_values": int(df.isnull().sum().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

def get_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a comprehensive descriptive statistical summary for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
        
    desc = num_df.describe().T
    desc['skewness'] = num_df.skew()
    desc['kurtosis'] = num_df.kurtosis()
    desc['missing_pct'] = (num_df.isnull().sum() / len(df)) * 100
    
    # Round for display
    return desc.round(2)

def get_categorical_summary(df: pd.DataFrame, max_cols: int = 10) -> dict:
    """Returns top categories and counts for categorical columns."""
    cat_df = df.select_dtypes(include=['object', 'category'])
    summary = {}
    
    for col in cat_df.columns[:max_cols]:
        val_counts = df[col].value_counts().head(5).to_dict()
        unique_count = df[col].nunique()
        summary[col] = {
            "unique_count": unique_count,
            "top_values": val_counts,
            "missing_pct": round((df[col].isnull().sum() / len(df)) * 100, 2)
        }
    return summary

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Returns Pearson correlation matrix for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    return num_df.corr().round(2)

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detects outlier counts per numeric column using Interquartile Range (IQR)."""
    num_df = df.select_dtypes(include=[np.number])
    outlier_report = []
    
    for col in num_df.columns:
        Q1 = num_df[col].quantile(0.25)
        Q3 = num_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = num_df[(num_df[col] < lower_bound) | (num_df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / len(df)) * 100, 2)
        
        outlier_report.append({
            "column": col,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        })
        
    return pd.DataFrame(outlier_report)
