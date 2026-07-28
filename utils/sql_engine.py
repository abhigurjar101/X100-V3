import duckdb
import pandas as pd
import streamlit as st

class SQLEngine:
    """
    Manages SQL execution against uploaded DataFrames using DuckDB (in-memory)
    or against Snowflake Cloud Data Warehouse if credentials are provided.
    """
    def __init__(self):
        self.duck_conn = duckdb.connect(database=':memory:', read_only=False)
        
    def register_dataframe(self, df: pd.DataFrame, table_name: str = "analytics_data"):
        """Registers a pandas DataFrame as a SQL table in DuckDB."""
        # DuckDB automatically registers global dataframes, but we explicitly register a view
        self.duck_conn.register(table_name, df)
        
    def execute_duckdb_query(self, query: str) -> tuple[pd.DataFrame | None, str | None]:
        """Executes a SQL query against the in-memory DuckDB database."""
        try:
            result_df = self.duck_conn.execute(query).df()
            return result_df, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def test_snowflake_connection(user, password, account, warehouse, database, schema):
        """
        Attempts to connect to Snowflake and verify access.
        If snowflake-connector-python is installed and credentials are valid, returns (True, message).
        """
        try:
            import snowflake.connector
            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema
            )
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()[0]
            conn.close()
            return True, f"Successfully connected to Snowflake! (Version: {version})"
        except ImportError:
            return False, "snowflake-connector-python package is not installed."
        except Exception as e:
            return False, f"Snowflake connection failed: {str(e)}"
            
    @staticmethod
    def execute_snowflake_query(user, password, account, warehouse, database, schema, query: str) -> tuple[pd.DataFrame | None, str | None]:
        """Executes a SQL query against Snowflake warehouse."""
        try:
            import snowflake.connector
            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema
            )
            cursor = conn.cursor()
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
            conn.close()
            return df, None
        except Exception as e:
            return None, str(e)
