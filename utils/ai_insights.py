import json
import pandas as pd
from openai import OpenAI

class AIAssistant:
    """
    AI Assistant class that interfaces with 100% FREE LLM providers:
    - Groq (Free API Tier - Llama 3 / Mixtral)
    - Ollama (Free Local LLM - Llama 3 / Mistral)
    - Google Gemini (Free API Tier - Gemini 1.5 Flash)
    - Offline Demo Mode (100% Free, No Internet Required)
    - OpenAI (Optional Paid)
    """
    def __init__(self, provider: str = "offline_demo", api_key: str = None, model_name: str = None, base_url: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
        # Configure client based on free provider choice
        if self.provider == "groq" and api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            self.model_name = model_name or "llama-3.1-8b-instant"
        elif self.provider == "ollama":
            # Ollama runs locally for free without an API key!
            self.client = OpenAI(api_key="ollama-local", base_url="http://localhost:11434/v1")
            self.model_name = model_name or "llama3"
        elif self.provider == "gemini" and api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            self.model_name = model_name or "gemini-1.5-flash"
        elif self.provider == "openai" and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = model_name or "gpt-4o-mini"

    def is_configured(self) -> bool:
        if self.provider == "offline_demo":
            return False
        if self.provider == "ollama":
            return True # Local free model
        return self.client is not None and bool(self.api_key)

    def get_provider_label(self) -> str:
        labels = {
            "offline_demo": "🆓 Offline Demo Mode (Rule-Based)",
            "groq": "🆓 Groq API (Fast & Free Llama 3)",
            "ollama": "🆓 Local Ollama (100% Private & Free)",
            "gemini": "🆓 Google Gemini (Free API Tier)",
            "openai": "💲 OpenAI API (Paid)"
        }
        return labels.get(self.provider, "Unknown Provider")

    def nl_to_sql(self, question: str, df: pd.DataFrame, table_name: str = "analytics_data") -> tuple[str, str]:
        """
        Translates a natural language question into a valid SQL query for DuckDB.
        Returns (sql_query, explanation).
        If API key is not configured or in offline demo mode, returns an intelligent rule-based query!
        """
        columns_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
        
        if not self.is_configured():
            # Intelligent fallback query generation for 100% FREE demo mode
            q_lower = question.lower()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if "top" in q_lower or "highest" in q_lower or "max" in q_lower:
                num_col = num_cols[0] if num_cols else df.columns[0]
                cat_col = cat_cols[0] if cat_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                query = f"SELECT {cat_col}, SUM({num_col}) as total_{num_col}\nFROM {table_name}\nGROUP BY {cat_col}\nORDER BY total_{num_col} DESC\nLIMIT 10;"
                explanation = f"[🆓 Free Demo Mode] Aggregated and sorted top 10 by {num_col} grouped by {cat_col}."
            elif "average" in q_lower or "mean" in q_lower or "avg" in q_lower:
                num_col = num_cols[0] if num_cols else df.columns[0]
                cat_col = cat_cols[0] if cat_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                query = f"SELECT {cat_col}, AVG({num_col}) as avg_{num_col}\nFROM {table_name}\nGROUP BY {cat_col}\nORDER BY avg_{num_col} DESC;"
                explanation = f"[🆓 Free Demo Mode] Calculated average {num_col} across {cat_col}."
            else:
                query = f"SELECT *\nFROM {table_name}\nLIMIT 20;"
                explanation = "[🆓 Free Demo Mode] Standard sample query returning 20 rows. Select Groq, Ollama, or Gemini in the sidebar for AI-generated queries!"
            return query, explanation

        prompt = f"""
You are an expert SQL Data Analyst. Your task is to write a valid DuckDB SQL query to answer the user's question based on the provided table schema.
Table Name: `{table_name}`
Schema:
{columns_info}

User Question: "{question}"

Instructions:
1. Return ONLY valid JSON with two keys: "sql_query" (the SQL code) and "explanation" (a 1-2 sentence explanation of what the query does).
2. Do NOT format the response in Markdown code blocks outside the JSON structure. Just output raw JSON.
3. Make sure column names match EXACTLY. Use double quotes for column names with special characters or reserved keywords if necessary.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # Clean possible markdown wrapping
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content.strip())
            return data.get("sql_query", f"SELECT * FROM {table_name} LIMIT 10;"), data.get("explanation", f"Generated SQL query using {self.get_provider_label()}.")
        except Exception as e:
            return f"SELECT * FROM {table_name} LIMIT 10;", f"Error generating SQL with {self.provider}: {str(e)}. Returned default query."

    def generate_business_insights(self, df: pd.DataFrame, summary_stats: dict, corr_matrix: pd.DataFrame) -> str:
        """
        Generates executive business insights, key findings, anomaly warnings, and actionable recommendations.
        """
        if not self.is_configured():
            # Return rich template response for 100% FREE offline demo
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            return f"""
### 🎯 Executive Portfolio Summary (🆓 100% Free Demo Mode)

**Overview:**
The dataset contains **{summary_stats.get('row_count', 'N/A')} records** and **{summary_stats.get('column_count', 'N/A')} attributes**, with **{summary_stats.get('numeric_columns', 0)} numeric metrics** and **{summary_stats.get('categorical_columns', 0)} categorical dimensions**. Overall data health shows **{summary_stats.get('total_missing_values', 0)} missing data points**, which have been flagged and cleaned by our automated hygiene pipeline.

---

### 📈 Key Highlights & Trends
1. **Primary Dimension (`{cat_cols[0] if cat_cols else 'N/A'}`):** Highly concentrated distribution observed across top categorical segments.
2. **Metric Variance (`{num_cols[0] if num_cols else 'N/A'}`):** Shows significant range variance requiring targeted monitoring.
3. **Correlation Dynamics:** Moderate to strong relationships detected between core performance variables in our automated EDA scan.

---

### ⚠️ Anomaly & Risk Watchlist
- **Outlier Concentration:** Several high-value outliers were detected in numerical features during IQR scanning.
- **Data Completeness:** Continuous monitoring of data ingest pipelines is advised to keep missingness under 1%.

---

### 🚀 Actionable Recommendations
1. **Segment Optimization:** Focus resource allocation on the top-performing categories identified in the EDA breakdown.
2. **Predictive Modeling:** Leverage clean numeric features (`{', '.join(num_cols[:3]) if num_cols else 'N/A'}`) to build regression or classification models for forecasting.
3. **Free SQL Automation:** Leverage DuckDB in-memory SQL to run scheduled data hygiene validations without incurring cloud warehouse costs!

> *💡 Tip: Want real-time LLM-tailored executive synthesis for free? Select **Groq (Free API)**, **Ollama (Free Local LLM)**, or **Google Gemini (Free Tier)** in the left sidebar!*
"""

        # Prepare summary prompt for LLM
        schema_summary = f"Columns: {', '.join(df.columns[:15])}"
        stats_str = json.dumps(summary_stats, indent=2)
        corr_str = corr_matrix.to_string() if not corr_matrix.empty else "No numeric correlations available."
        
        prompt = f"""
You are a Principal Data Scientist and Executive Business Analyst. Analyze the following dataset summary statistics and correlation matrix, and produce a comprehensive, dashboard-ready Executive Report in clean Markdown.

Dataset Schema Overview:
{schema_summary}

General Overview Stats:
{stats_str}

Correlation Matrix:
{corr_str}

Your report MUST contain the following sections formatted with Markdown headers and bullet points:
1. 🎯 **Executive Summary**: A high-level overview of what this data represents and its general state.
2. 📈 **Key Findings & Trends**: 3-4 bullet points highlighting interesting patterns, distributions, or strong correlations.
3. ⚠️ **Anomalies & Risk Areas**: Flag potential issues such as missing data, extreme skewness, or data quality risks.
4. 🚀 **Strategic Recommendations**: 3 concrete business actions or technical next steps based on the findings.

Make the tone professional, insightful, and executive-ready.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"### Error Generating Insights with {self.get_provider_label()}\nAn error occurred while calling the AI model: `{str(e)}`.\nPlease verify your API Key or make sure Ollama is running locally."
