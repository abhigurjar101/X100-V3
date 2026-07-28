import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Custom luxury dark color palette
DARK_PALETTE = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185", "#34d399", "#fbbf24"]

def apply_custom_theme(fig):
    """Applies luxury SaaS dark theme styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        colorway=DARK_PALETTE,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    )
    return fig

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = None, color_col: str = None):
    """Creates a stylized horizontal or vertical bar chart."""
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col if color_col and color_col in df.columns else None,
        title=title or f"{y_col} by {x_col}",
        color_discrete_sequence=DARK_PALETTE
    )
    return apply_custom_theme(fig)

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = None, color_col: str = None):
    """Creates a stylized line/time-series chart."""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col if color_col and color_col in df.columns else None,
        title=title or f"{y_col} Trend over {x_col}",
        color_discrete_sequence=DARK_PALETTE,
        markers=True
    )
    return apply_custom_theme(fig)

def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, size_col: str = None, color_col: str = None, title: str = None):
    """Creates an interactive scatter plot."""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        size=size_col if size_col and size_col in df.columns else None,
        color=color_col if color_col and color_col in df.columns else None,
        title=title or f"{y_col} vs {x_col}",
        color_discrete_sequence=DARK_PALETTE,
        opacity=0.85
    )
    return apply_custom_theme(fig)

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str = None):
    """Creates a styled donut/pie chart."""
    fig = px.pie(
        df, 
        names=names_col, 
        values=values_col, 
        title=title or f"Proportion of {values_col} by {names_col}",
        color_discrete_sequence=DARK_PALETTE,
        hole=0.45
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return apply_custom_theme(fig)

def create_histogram(df: pd.DataFrame, col: str, bins: int = 30, title: str = None):
    """Creates a histogram for feature distribution analysis."""
    fig = px.histogram(
        df, 
        x=col, 
        nbins=bins, 
        title=title or f"Distribution of {col}",
        color_discrete_sequence=["#38bdf8"],
        opacity=0.8,
        marginal="box"
    )
    return apply_custom_theme(fig)

def create_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Feature Correlation Heatmap"):
    """Creates an interactive correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='IceFire',
        zmin=-1, zmax=1,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        hoverinfo="x+y+z"
    ))
    fig.update_layout(title=title)
    return apply_custom_theme(fig)
