import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class AutoMLModeler:
    """
    100% Free Open-Source AutoML Engine using Scikit-Learn.
    Automatically handles categorical encoding, trains Random Forest models,
    evaluates performance, and extracts feature importances.
    """
    @staticmethod
    def prepare_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list, str]:
        """Prepares features and target, dropping IDs and dates, and one-hot encoding categoricals."""
        df_clean = df.copy().dropna(subset=[target_col])
        
        # Identify target type
        target_series = df_clean[target_col]
        is_classification = False
        if target_series.dtype == 'object' or str(target_series.dtype).startswith('category') or target_series.nunique() <= 10:
            is_classification = True
            
        # Drop columns that look like IDs or dates
        drop_cols = [target_col]
        for col in df_clean.columns:
            if col != target_col:
                if 'date' in col.lower() or 'id' in col.lower() or df_clean[col].dtype == 'datetime64[ns]':
                    drop_cols.append(col)
                elif df_clean[col].nunique() == len(df_clean): # Unique ID column
                    drop_cols.append(col)
                    
        X_raw = df_clean.drop(columns=list(set(drop_cols)), errors='ignore')
        
        # One-hot encode categorical features
        X = pd.get_dummies(X_raw, drop_first=True)
        
        # Handle target encoding if classification
        y = target_series
        if is_classification and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
            
        return X, y, X.columns.tolist(), "classification" if is_classification else "regression"

    @classmethod
    def train_and_evaluate(cls, df: pd.DataFrame, target_col: str) -> dict:
        """Trains Random Forest model and returns evaluation metrics and feature importances."""
        try:
            X, y, feature_names, task_type = cls.prepare_data(df, target_col)
            
            if X.empty or len(feature_names) == 0:
                return {"error": "Not enough features available to train a predictive model."}
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                metric_name = "Model Accuracy"
                score_str = f"{round(score * 100, 2)}%"
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                metric_name = "R² Variance Explained"
                score_str = f"{round(score * 100, 2)}%"
                
            # Extract Feature Importances
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)
            
            # Convert importance to percentage
            imp_df["Importance_Pct"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(2)
            
            return {
                "status": "success",
                "task_type": task_type.capitalize(),
                "metric_name": metric_name,
                "score_str": score_str,
                "feature_count": len(feature_names),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "importance_df": imp_df
            }
        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"}
