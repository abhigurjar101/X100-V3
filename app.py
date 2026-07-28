import streamlit as st
import pandas as pd
import numpy as np
import os

# Import local utilities
from utils.cleaner import run_full_pipeline
from utils.eda import (
    get_summary_stats, 
    get_descriptive_table, 
    get_categorical_summary, 
    get_correlation_matrix, 
    detect_outliers_iqr
)
from utils.sql_engine import SQLEngine
from utils.ai_insights import AIAssistant
from utils.visualizer import (
    create_bar_chart, 
    create_line_chart, 
    create_scatter_chart, 
    create_pie_chart, 
    create_histogram, 
    create_correlation_heatmap
)
from utils.ml_engine import AutoMLModeler

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst Assistant (100% Free Stack)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None
if "sql_engine" not in st.session_state:
    st.session_state.sql_engine = SQLEngine()
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "active_sql_query" not in st.session_state:
    st.session_state.active_sql_query = "SELECT * FROM analytics_data LIMIT 10;"

# Sidebar Configuration
with st.sidebar:
    st.title("⚡ AI Analyst Core")
    st.caption("💎 Ultra-Luxury 2026 SaaS • 100% Free Stack")
    st.markdown("---")
    
    st.subheader("🤖 100% Free AI Provider")
    provider_choice = st.selectbox(
        "Select AI Engine:",
        [
            "🆓 Offline Demo Mode (Rule-Based)",
            "🆓 Groq API (Fast Free Llama 3)",
            "🆓 Local Ollama (100% Private & Free)",
            "🆓 Google Gemini (Free Tier)",
            "💲 OpenAI API (Paid Option)"
        ],
        index=0,
        help="Choose a completely free option! Offline Demo requires no internet or key. Groq and Gemini offer generous free API tiers. Ollama runs locally on your laptop!"
    )
    
    # Parse provider name
    provider_map = {
        "🆓 Offline Demo Mode (Rule-Based)": "offline_demo",
        "🆓 Groq API (Fast Free Llama 3)": "groq",
        "🆓 Local Ollama (100% Private & Free)": "ollama",
        "🆓 Google Gemini (Free Tier)": "gemini",
        "💲 OpenAI API (Paid Option)": "openai"
    }
    selected_provider = provider_map[provider_choice]
    
    api_key = None
    model_name = None
    
    if selected_provider == "groq":
        api_key = st.text_input("Groq Free API Key (console.groq.com)", type="password", help="Get a free key from console.groq.com")
        model_name = st.selectbox("Groq Model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    elif selected_provider == "ollama":
        st.info("💡 Ollama runs locally at http://localhost:11434. No API Key required!")
        model_name = st.text_input("Ollama Model Name", value="llama3", help="Make sure you ran 'ollama run llama3' in terminal")
    elif selected_provider == "gemini":
        api_key = st.text_input("Google Gemini API Key (aistudio.google.com)", type="password", help="Get a free key from Google AI Studio")
        model_name = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    elif selected_provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])
    else:
        st.caption("💡 Running in instant 100% Free Demo Mode! No API key, internet, or setup needed.")

    # Initialize AI Assistant
    ai_assistant = AIAssistant(provider=selected_provider, api_key=api_key, model_name=model_name)
    if ai_assistant.is_configured():
        st.success(f"Connected: {ai_assistant.get_provider_label()}")
    elif selected_provider == "offline_demo":
        st.info("✅ Instant Demo Mode Active")
    else:
        st.warning("⚠️ Enter API Key above to activate this AI engine.")
        
    st.markdown("---")
    st.subheader("❄️ Snowflake / SQL Engine")
    st.markdown("💡 **Default Engine:** `DuckDB` (100% Free In-Memory SQL)")
    with st.expander("Configure Live Snowflake (Optional)"):
        sf_user = st.text_input("User")
        sf_password = st.text_input("Password", type="password")
        sf_account = st.text_input("Account Identifier")
        sf_warehouse = st.text_input("Warehouse", value="COMPUTE_WH")
        sf_db = st.text_input("Database", value="ANALYTICS_DB")
        sf_schema = st.text_input("Schema", value="PUBLIC")
        
        if st.button("Test Snowflake Connection"):
            success, msg = SQLEngine.test_snowflake_connection(sf_user, sf_password, sf_account, sf_warehouse, sf_db, sf_schema)
            if success:
                st.success(msg)
            else:
                st.error(msg)
                
    st.markdown("---")
    st.markdown("### 📚 100% Free Tech Stack:")
    st.markdown("`Python` `DuckDB` `Scikit-Learn` `Streamlit` `Plotly` `Ollama` `Groq`")

# Main Header
st.title("🤖 AI-Powered Data Analyst Assistant")
st.markdown("Automate messy data cleaning, conduct deep exploratory data analysis, generate natural-language SQL queries, train predictive machine learning models, and synthesize executive insights — **using an ultra-luxury, 100% Free & Open-Source Tech Stack!**")

# Create Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 1. Data Upload & Auto-Clean", 
    "📊 2. Automated EDA", 
    "💬 3. AI SQL Studio", 
    "📈 4. Chart Creator", 
    "💡 5. Executive Insights",
    "🤖 6. Predictive AutoML"
])

# ---------------------------------------------------------
# TAB 1: DATA UPLOAD & AUTO-CLEAN
# ---------------------------------------------------------
with tab1:
    st.subheader("📁 Ingest & Clean Dataset")
    st.markdown("Upload a messy spreadsheet or CSV file. The AI Analyst automatically standardizes column names, imputes missing values, and removes duplicate entries using free local computation.")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel spreadsheet", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.raw_df = df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            
    # Sample dataset option if no file uploaded
    if st.session_state.raw_df is None:
        st.info("💡 Don't have a file right now? Click below to load our sample SaaS Customer Analytics dataset to explore right now!")
        if st.button("🌟 Load Sample SaaS Analytics Dataset", use_container_width=True):
            np.random.seed(42)
            sample_data = {
                "Customer ID": [f"CUST_{i:04d}" for i in range(1, 201)],
                "Company Name ": [f"Enterprise_{i}" if i % 2 == 0 else f"Startup_{i}" for i in range(1, 201)],
                "Monthly_Recurring_Revenue ($)": np.random.normal(5000, 2000, 200).round(2),
                "Subscription Plan": np.random.choice(["Starter", "Professional", "Enterprise", np.nan], 200, p=[0.4, 0.4, 0.15, 0.05]),
                "Customer Support Tickets": np.random.poisson(2, 200),
                "Churn_Risk_Score": np.random.uniform(0.01, 0.99, 200).round(2),
                "Sign-Up Date": pd.date_range(start="2024-01-01", periods=200, freq="D").astype(str),
                "Region": np.random.choice(["North America", "EMEA", "APAC", "LATAM"], 200)
            }
            # Inject duplicates and missing values for realism
            df_sample = pd.DataFrame(sample_data)
            df_sample.loc[10:15, "Monthly_Recurring_Revenue ($)"] = np.nan
            df_sample = pd.concat([df_sample, df_sample.iloc[:5]], ignore_index=True)
            st.session_state.raw_df = df_sample
            st.rerun()

    if st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        st.markdown("### Raw Dataset Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Rows", len(raw_df))
        col2.metric("Initial Columns", len(raw_df.columns))
        col3.metric("Missing Data Points", int(raw_df.isnull().sum().sum()))
        
        st.markdown("---")
        st.subheader("⚙️ Automated Hygiene Pipeline")
        
        impute_method = st.selectbox(
            "Missing Value Imputation Strategy",
            ["auto", "median", "mean", "mode", "drop"],
            help="'auto' imputes numerical nulls with the median and categorical nulls with the mode."
        )
        
        if st.button("🚀 Run Auto-Cleaner & Pipeline", use_container_width=True):
            with st.spinner("Standardizing names, casting data types, and imputing nulls..."):
                clean_df, report = run_full_pipeline(raw_df, impute_strategy=impute_method)
                st.session_state.clean_df = clean_df
                st.session_state.cleaning_report = report
                # Register in DuckDB
                st.session_state.sql_engine.register_dataframe(clean_df, "analytics_data")
            st.success("🎉 Dataset successfully cleaned and registered into in-memory DuckDB SQL engine!")
            
        if st.session_state.clean_df is not None:
            clean_df = st.session_state.clean_df
            report = st.session_state.cleaning_report
            
            st.markdown("### 💎 Cleaned Dataset Preview")
            st.dataframe(clean_df.head(10), use_container_width=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cleaned Rows", len(clean_df), delta=f"{len(clean_df) - len(raw_df)} dups removed")
            c2.metric("Cleaned Columns", len(clean_df.columns))
            c3.metric("Remaining Nulls", int(clean_df.isnull().sum().sum()))
            c4.metric("Memory Usage", f"{round(clean_df.memory_usage(deep=True).sum() / 1024, 1)} KB")
            
            with st.expander("🔍 View Detailed Pipeline Report"):
                st.json(report)

# ---------------------------------------------------------
# TAB 2: AUTOMATED EDA
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please upload and clean a dataset in Tab 1 first!")
    else:
        df = st.session_state.clean_df
        stats = get_summary_stats(df)
        
        st.markdown("#### General Overview")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Total Records", stats["row_count"])
        g2.metric("Numeric Features", stats["numeric_columns"])
        g3.metric("Categorical Features", stats["categorical_columns"])
        g4.metric("Datetime Features", stats["datetime_columns"])
        
        st.markdown("---")
        st.markdown("#### Descriptive Statistics (Numerical Attributes)")
        desc_table = get_descriptive_table(df)
        if not desc_table.empty:
            st.dataframe(desc_table, use_container_width=True)
        else:
            st.info("No numerical columns found for statistical summary.")
            
        st.markdown("---")
        st.markdown("#### Categorical Segment Analysis")
        cat_summary = get_categorical_summary(df)
        if cat_summary:
            cat_cols = list(cat_summary.keys())
            selected_cat = st.selectbox("Select Categorical Feature to Inspect", cat_cols)
            if selected_cat:
                col_data = cat_summary[selected_cat]
                c_a, c_b = st.columns([1, 2])
                with c_a:
                    st.metric("Unique Categories", col_data["unique_count"])
                    st.metric("Missing %", f"{col_data['missing_pct']}%")
                with c_b:
                    st.markdown("**Top 5 Most Frequent Values:**")
                    top_vals_df = pd.DataFrame(list(col_data["top_values"].items()), columns=["Category", "Count"])
                    st.dataframe(top_vals_df, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns found.")
            
        st.markdown("---")
        st.markdown("#### 🚨 Outlier Detection (Interquartile Range Scan)")
        outlier_df = detect_outliers_iqr(df)
        if not outlier_df.empty:
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            total_outliers = outlier_df["outlier_count"].sum()
            if total_outliers > 0:
                st.markdown(f"<div class='insight-box'>⚠️ <b>System Notice:</b> Detected a total of <b>{total_outliers} statistical outliers</b> across numerical attributes. Consider filtering or capping these values before running predictive modeling.</div>", unsafe_allow_html=True)
        else:
            st.info("No numerical columns available for IQR outlier detection.")

# ---------------------------------------------------------
# TAB 3: AI SQL STUDIO (OVERHAULED & 100% INTERACTIVE)
# ---------------------------------------------------------
with tab3:
    st.subheader("💬 Natural Language to SQL Studio")
    st.markdown(f"Ask complex analytical questions in plain English using **{ai_assistant.get_provider_label()}** or use our **1-Click Suggestion Pills** below! The AI translates your question into high-performance SQL, executes it against the free in-memory DuckDB engine, and displays the results.")
    
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please upload and clean a dataset in Tab 1 first!")
    else:
        df = st.session_state.clean_df
        sql_engine = st.session_state.sql_engine
        
        # Explicitly register dataframe on every rerun to guarantee DuckDB view exists
        sql_engine.register_dataframe(df, "analytics_data")
        
        st.markdown("#### 💡 1-Click Quick Suggestion Queries:")
        q1, q2, q3, q4, q5 = st.columns(5)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        primary_num = num_cols[0] if num_cols else df.columns[0]
        primary_cat = cat_cols[0] if cat_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        if q1.button("🏆 Top 10 by Revenue", use_container_width=True):
            st.session_state.active_sql_query = f"SELECT {primary_cat}, SUM({primary_num}) AS total_{primary_num}\nFROM analytics_data\nGROUP BY {primary_cat}\nORDER BY total_{primary_num} DESC\nLIMIT 10;"
        if q2.button("📊 Average by Region", use_container_width=True):
            st.session_state.active_sql_query = f"SELECT {primary_cat}, AVG({primary_num}) AS avg_{primary_num}\nFROM analytics_data\nGROUP BY {primary_cat}\nORDER BY avg_{primary_num} DESC;"
        if q3.button("🚨 Highest Risk Sample", use_container_width=True):
            risk_col = "churn_risk_score" if "churn_risk_score" in df.columns else primary_num
            st.session_state.active_sql_query = f"SELECT *\nFROM analytics_data\nORDER BY {risk_col} DESC\nLIMIT 10;"
        if q4.button("📈 Count by Segment", use_container_width=True):
            st.session_state.active_sql_query = f"SELECT {primary_cat}, COUNT(*) as total_customers\nFROM analytics_data\nGROUP BY {primary_cat}\nORDER BY total_customers DESC;"
        if q5.button("📋 Sample 15 Rows", use_container_width=True):
            st.session_state.active_sql_query = "SELECT * FROM analytics_data LIMIT 15;"
            
        st.markdown("---")
        
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            nl_query = st.text_input(
                "Or ask your own custom question in plain English:",
                placeholder=f"e.g., Show the top 5 {primary_cat} by average {primary_num}"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("⚡ Generate AI SQL", use_container_width=True)
            
        if generate_btn and nl_query:
            with st.spinner(f"Translating via {ai_assistant.get_provider_label()}..."):
                sql_code, explanation = ai_assistant.nl_to_sql(nl_query, df, table_name="analytics_data")
                st.session_state.active_sql_query = sql_code
                st.success(f"💡 AI Explanation: {explanation}")
                
        st.markdown("#### 🛠️ Interactive SQL Query Editor")
        st.caption("You can view, tweak, or write raw SQL directly below and click Run to test!")
        
        edited_sql = st.text_area("DuckDB SQL Code:", value=st.session_state.active_sql_query, height=140)
        st.session_state.active_sql_query = edited_sql
        
        if st.button("▶️ Run / Re-run SQL Query", use_container_width=True):
            with st.spinner("Executing SQL against in-memory DuckDB engine..."):
                res_df, err = sql_engine.execute_duckdb_query(edited_sql)
                if err:
                    st.error(f"❌ SQL Execution Error: {err}\n\n*Tip: Check that column names match your clean dataframe preview in Tab 1!*")
                elif res_df is not None:
                    st.markdown("#### ⚡ Query Execution Results")
                    st.dataframe(res_df, use_container_width=True)
                    
                    if not res_df.empty:
                        csv_export = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results (CSV)", data=csv_export, file_name="query_results.csv", mime="text/csv")
                else:
                    st.info("Query executed successfully, but returned 0 rows.")

# ---------------------------------------------------------
# TAB 4: CHART CREATOR
# ---------------------------------------------------------
with tab4:
    st.subheader("📈 Dynamic Chart & Dashboard Creator")
    st.markdown("Generate interactive, luxury SaaS Plotly visualizations tailored for presentation decks and Power BI dashboards.")
    
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please upload and clean a dataset in Tab 1 first!")
    else:
        df = st.session_state.clean_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        chart_type = st.selectbox(
            "Select Visualization Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie/Donut Chart", "Feature Distribution (Histogram)", "Correlation Heatmap"]
        )
        
        st.markdown("---")
        
        if chart_type == "Bar Chart":
            c_x, c_y, c_color = st.columns(3)
            x_col = c_x.selectbox("X-Axis (Category)", all_cols, index=0)
            y_col = c_y.selectbox("Y-Axis (Metric)", num_cols, index=0 if num_cols else None)
            color_col = c_color.selectbox("Color Grouping (Optional)", ["None"] + cat_cols, index=0)
            
            if st.button("📊 Render Bar Chart", use_container_width=True):
                color_arg = None if color_col == "None" else color_col
                fig = create_bar_chart(df, x_col, y_col, color_col=color_arg)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Line Chart":
            c_x, c_y, c_color = st.columns(3)
            x_col = c_x.selectbox("X-Axis (Time/Sequence)", all_cols, index=0)
            y_col = c_y.selectbox("Y-Axis (Metric)", num_cols, index=0 if num_cols else None)
            color_col = c_color.selectbox("Color Grouping (Optional)", ["None"] + cat_cols, index=0)
            
            if st.button("📈 Render Line Chart", use_container_width=True):
                color_arg = None if color_col == "None" else color_col
                fig = create_line_chart(df, x_col, y_col, color_col=color_arg)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Scatter Plot":
            c_x, c_y, c_size, c_color = st.columns(4)
            x_col = c_x.selectbox("X-Axis", num_cols, index=0 if num_cols else None)
            y_col = c_y.selectbox("Y-Axis", num_cols, index=1 if len(num_cols) > 1 else 0)
            size_col = c_size.selectbox("Bubble Size (Optional)", ["None"] + num_cols, index=0)
            color_col = c_color.selectbox("Color Grouping (Optional)", ["None"] + cat_cols, index=0)
            
            if st.button("🔵 Render Scatter Plot", use_container_width=True):
                size_arg = None if size_col == "None" else size_col
                color_arg = None if color_col == "None" else color_col
                fig = create_scatter_chart(df, x_col, y_col, size_col=size_arg, color_col=color_arg)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Pie/Donut Chart":
            c_names, c_vals = st.columns(2)
            names_col = c_names.selectbox("Category Dimension", cat_cols if cat_cols else all_cols, index=0)
            values_col = c_vals.selectbox("Values Metric", num_cols, index=0 if num_cols else None)
            
            if st.button("🍩 Render Donut Chart", use_container_width=True):
                fig = create_pie_chart(df, names_col, values_col)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Feature Distribution (Histogram)":
            c_col, c_bins = st.columns(2)
            dist_col = c_col.selectbox("Select Attribute", num_cols if num_cols else all_cols, index=0)
            bins = c_bins.slider("Number of Bins", min_value=10, max_value=100, value=30)
            
            if st.button("📉 Render Histogram", use_container_width=True):
                fig = create_histogram(df, dist_col, bins=bins)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Correlation Heatmap":
            st.markdown("Visualizing Pearson correlation relationships across all numerical attributes.")
            if st.button("🔥 Render Heatmap", use_container_width=True):
                corr_matrix = get_correlation_matrix(df)
                if not corr_matrix.empty:
                    fig = create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Not enough numerical columns to calculate correlation matrix.")

# ---------------------------------------------------------
# TAB 5: EXECUTIVE INSIGHTS
# ---------------------------------------------------------
with tab5:
    st.subheader("💡 Executive Business Insights & Synthesis")
    st.markdown(f"Let **{ai_assistant.get_provider_label()}** synthesize the dataset statistics, identify key trends, flag anomaly risks, and formulate executive-ready strategic recommendations.")
    
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please upload and clean a dataset in Tab 1 first!")
    else:
        df = st.session_state.clean_df
        stats = get_summary_stats(df)
        corr = get_correlation_matrix(df)
        
        col_synth, col_export = st.columns(2)
        with col_synth:
            generate_report = st.button("✨ Generate AI Executive Report", use_container_width=True)
        with col_export:
            # Standalone HTML Report Generation
            html_content = f"""
            <html>
            <head><title>AI Analyst Portfolio Dashboard Report</title>
            <style>
                body {{ font-family: Inter, sans-serif; background: #050811; color: #f1f5f9; padding: 40px; }}
                h1, h2 {{ color: #38bdf8; font-family: Outfit, sans-serif; }}
                .metric-box {{ background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #38bdf8; margin: 10px 0; }}
            </style>
            </head>
            <body>
                <h1>🤖 Executive Analytics Portfolio Briefing</h1>
                <p>Generated by 100% Free Open-Source AI Data Analyst Assistant</p>
                <hr>
                <h2>📊 Dataset Overview Metrics</h2>
                <div class="metric-box">
                    <b>Total Records:</b> {stats['row_count']} | <b>Features:</b> {stats['column_count']} | <b>Numeric Features:</b> {stats['numeric_columns']}
                </div>
                <h2>📈 Key Statistical Breakdown</h2>
                <pre style="background: #090e1f; padding: 15px; border-radius: 12px; color: #38bdf8; border: 1px solid #334155;">{get_descriptive_table(df).to_string()}</pre>
                <hr>
                <p><i>Note: For interactive SQL queries and Plotly visualizations, view the live Streamlit dashboard!</i></p>
            </body>
            </html>
            """
            st.download_button(
                "📥 Export Dashboard Report (HTML)",
                data=html_content.encode('utf-8'),
                file_name="portfolio_analytics_report.html",
                mime="text/html",
                use_container_width=True
            )

        if generate_report:
            with st.spinner(f"Synthesizing statistics and formulating executive recommendations via {ai_assistant.get_provider_label()}..."):
                report_md = ai_assistant.generate_business_insights(df, stats, corr)
                
            st.markdown("---")
            st.markdown(report_md)
            
            # Export report option
            st.download_button(
                "📥 Download AI Executive Summary (Markdown)",
                data=report_md.encode('utf-8'),
                file_name="executive_report.md",
                mime="text/markdown"
            )

# ---------------------------------------------------------
# TAB 6: PREDICTIVE AUTOML & FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab6:
    st.subheader("🤖 Predictive Modeling & Feature Importance (Scikit-Learn)")
    st.markdown("Bridge Data Analytics and Machine Learning! Select a target variable to predict, and our **100% Free Open-Source AutoML Engine** automatically trains a Random Forest model, calculates variance/accuracy, and ranks the most influential drivers.")
    
    if st.session_state.clean_df is None:
        st.warning("⚠️ Please upload and clean a dataset in Tab 1 first!")
    else:
        df = st.session_state.clean_df
        target_candidate = st.selectbox("Select Target Variable to Predict:", df.columns.tolist())
        
        if st.button("🚀 Train Predictive Model & Rank Drivers", use_container_width=True):
            with st.spinner(f"Encoding categorical features and training Random Forest on '{target_candidate}'..."):
                results = AutoMLModeler.train_and_evaluate(df, target_candidate)
                
            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"🎉 Model Successfully Trained! ({results['task_type']} Engine)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric(results["metric_name"], results["score_str"])
                m2.metric("Features Analyzed", results["feature_count"])
                m3.metric("Train / Test Samples", f"{results['train_samples']} / {results['test_samples']}")
                
                st.markdown("---")
                st.markdown(f"#### 🏆 Top Predictive Drivers for `{target_candidate}`")
                
                imp_df = results["importance_df"]
                fig = create_bar_chart(imp_df, "Importance_Pct", "Feature", title=f"Feature Importance Ranking (%) for {target_candidate}")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 View Raw Feature Importance Table"):
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)
