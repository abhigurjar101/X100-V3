# 🤖 AI-Powered Data Analyst Assistant

An experimental Data Science & Analytics web application built with **Streamlit**, **DuckDB**, **Scikit-Learn**, **Plotly**, and **LLMs (Groq, Gemini, Ollama, OpenAI)**.

This tool automates messy spreadsheet cleaning, performs comprehensive Exploratory Data Analysis (EDA), translates natural language questions into high-performance SQL, trains predictive machine learning models, and synthesizes executive business briefings.

---

## 🌟 Features & 6-Tab Workspace

1. **📁 Auto-Clean & Data Hygiene:** Upload CSV or Excel files. Automatically standardize column names, impute missing values (`median`, `mean`, `mode`, `auto`), format dates, and remove duplicates. Includes a built-in realistic SaaS dataset for demoing!
2. **📊 Automated EDA Engine:** Instant overview metrics, descriptive statistical tables, categorical frequency breakdowns, and an Interquartile Range (IQR) outlier detection scanner.
3. **💬 AI SQL Studio:** Ask analytical questions in plain English. Generates valid DuckDB SQL dynamically, executes against an in-memory database, and lets you download results as CSV. Supports 5 pluggable AI engines:
   - 🆓 **Offline Demo Mode** (Rule-based fallback, zero internet/key required)
   - 🆓 **Groq API** (Hosted model API; availability and pricing depend on provider)
   - 🆓 **Local Ollama** (100% private & free local LLM execution)
   - 🆓 **Google Gemini** (Hosted model API; availability and pricing depend on provider)
   - 💲 **OpenAI API** (Optional paid model support)
4. **📈 Dynamic Chart Creator:** Generate Bar Charts, Line Charts, Scatter Plots, Donut Charts, Histograms, and Correlation Heatmaps styled in a custom dark SaaS neon colorway.
5. **💡 Executive Business Insights & Export:** Generate structured Markdown executive reports highlighting trends, anomaly risks, and strategic recommendations. Export a standalone styled HTML briefing dashboard report in one click!
6. **🤖 Predictive AutoML & Feature Importance:** Select any target column. The engine automatically detects classification vs. regression, one-hot encodes categoricals, trains a Random Forest model using Scikit-Learn, evaluates variance/accuracy, and ranks the top 15 predictive drivers!

---

## 🚀 Tech Stack
* **Frontend & UI:** `Streamlit`, `Custom CSS`
* **In-Memory SQL:** `DuckDB`
* **Machine Learning & EDA:** `Scikit-Learn`, `Pandas`, `NumPy`
* **Data Visualization:** `Plotly Graph Objects & Express`
* **AI / LLM Integration:** `Groq`, `Google Gemini`, `Ollama`, `OpenAI`
* **Cloud Warehouse Ready:** `Snowflake Connector` interface

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhigurjar101/X100-V3.git
   cd X100-V3
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 🌐 Optional deployment (Streamlit Community Cloud)

For an optional hosted demo, check the hosting provider’s current plans and limits:
1. Push this folder to a public repository on **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**, select your repository, branch, and set the Main file path to `app.py`.
4. Click **"Deploy!"**
5. *(Optional)* If you want to hardcode your free API keys so users don't have to type them in the sidebar, go to your app settings on Streamlit Cloud -> **Advanced Settings -> Secrets** and add:
   ```toml
   GROQ_API_KEY = "your-groq-key-here"
   GEMINI_API_KEY = "your-gemini-key-here"
   ```

## Evaluation boundary

This is an exploratory analytics application. AutoML reports metrics for the supplied dataset; no general accuracy, latency or business-impact benchmark is established here. Review preprocessing leakage, task selection and dataset size before interpreting model scores. Provider calls can send prompts and data to the selected service and may incur charges. The Snowflake helpers require a separately configured account.

The repository currently includes tracked Python bytecode and lacks a focused test suite. Those are cleanup priorities; the README does not claim production readiness.
