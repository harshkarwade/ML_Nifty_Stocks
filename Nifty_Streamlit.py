import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- Technical Indicator Function from the Notebook ---

def calculate_rsi(data, window=50):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_and_preprocess_data():
    """
    Loads the raw data and recreates the necessary preprocessing steps 
    (Date conversion, SMA, RSI calculation, and filling NaNs) 
    from the original ML_Project.ipynb notebook.
    """
    try:
        # NOTE FOR USER: Ensure 'Nifty_Stocks.csv' is accessible.
        df = pd.read_csv("Nifty_Stocks.csv") 
    except FileNotFoundError:
        st.error("🚨 Error: 'Nifty_Stocks.csv' not found. Please ensure the file is in the correct directory.")
        st.stop()

    # Data Cleaning and Feature Engineering (from Cells 7, 9, 12, 15, 16, 22, 23)
    df = df.fillna(0)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate SMA_50, SMA_200, and RSI for each stock symbol
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean().fillna(0))
    df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean().fillna(0))
    df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x, window=50).fillna(0))
    
    return df.sort_values(by='Date')

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Stock Price Comparison Tool")
    st.title("📊 Stock Price Pattern Comparison Tool")
    st.markdown("Select one or more stocks and a date range below to compare their price movements.")
    st.markdown("---")

    # Load data
    df_full = load_and_preprocess_data()
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Filter & Analysis Options")

    # 1. Select Multiple Symbols for Comparison
    all_symbols = sorted(df_full['Symbol'].unique())
    selected_symbols = st.sidebar.multiselect(
        "Select Stock Symbols (for comparison)", 
        all_symbols, 
        default=all_symbols[:2] if len(all_symbols) >= 2 else all_symbols
    )

    if not selected_symbols:
        st.warning("Please select at least one stock symbol to visualize.")
        return

    # Filter data for selected symbols
    df_filtered_symbols = df_full[df_full['Symbol'].isin(selected_symbols)].reset_index(drop=True)

    if df_filtered_symbols.empty:
        st.error("No data available for the selected symbols.")
        return

    # 2. Select Date Range (Based on the min/max date of the selected subset)
    min_date = df_filtered_symbols['Date'].min().to_pydatetime().date()
    max_date = df_filtered_symbols['Date'].max().to_pydatetime().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_final = df_filtered_symbols[
            (df_filtered_symbols['Date'] >= pd.to_datetime(start_date)) & 
            (df_filtered_symbols['Date'] <= pd.to_datetime(end_date))
        ]
    else:
        st.warning("Please select a valid start and end date for the visualization.")
        df_final = pd.DataFrame() # Empty dataframe if date selection is incomplete
        return

    # --- Main Content: Visualization ---
    
    if not df_final.empty:
        st.subheader(f"Close Price Comparison: {', '.join(selected_symbols)}")
        
        # Initialize a single Plotly figure for price comparison
        fig = go.Figure()

        # Iterate over selected symbols and add a trace for each
        for symbol in selected_symbols:
            df_stock = df_final[df_final['Symbol'] == symbol]
            
            # Add Close Price trace
            fig.add_trace(go.Scatter(
                x=df_stock['Date'], 
                y=df_stock['Close'], 
                mode='lines', 
                name=f'{symbol} Close Price'
            ))

        # Update layout and titles for the comparison chart
        fig.update_layout(
            height=600, 
            title_text="Stock Price Pattern Comparison",
            xaxis_title="Date",
            yaxis_title="Close Price (INR)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Displaying individual indicator for the FIRST selected stock ---
        # It's generally messy to compare RSI/SMA across different stocks due to scaling, 
        # so we show them for the primary selected stock for detailed analysis.
        if selected_symbols:
            st.subheader(f"Technical Indicators for Detailed Analysis: {selected_symbols[0]}")
            df_single = df_final[df_final['Symbol'] == selected_symbols[0]]
            
            fig_detail = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.7, 0.3], 
                                subplot_titles=[f"{selected_symbols[0]} Price & Moving Averages", "Relative Strength Index (RSI)"])
            
            # Row 1: Price and SMAs
            fig_detail.add_trace(go.Scatter(x=df_single['Date'], y=df_single['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)
            fig_detail.add_trace(go.Scatter(x=df_single['Date'], y=df_single['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', dash='dash')), row=1, col=1)
            fig_detail.add_trace(go.Scatter(x=df_single['Date'], y=df_single['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', dash='dot')), row=1, col=1)
            fig_detail.update_yaxes(title_text="Price (INR)", row=1, col=1)
            
            # Row 2: RSI
            fig_detail.add_trace(go.Scatter(x=df_single['Date'], y=df_single['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
            fig_detail.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
            fig_detail.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
            fig_detail.update_yaxes(title_text="RSI (50-Day)", range=[0, 100], row=2, col=1)
            fig_detail.update_xaxes(title_text="Date", row=2, col=1)
            fig_detail.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)

            st.plotly_chart(fig_detail, use_container_width=True)


    st.markdown("---")
    
    # --- 3. Regression Model Summary ---
    st.header("Regression Model Performance Summary (Whole Dataset)")
    st.write("The notebook trained three regression models to predict the 'Close' price across the **entire dataset**. The high R² scores indicate excellent predictive power based on the features used.")

    # Results from the notebook (Cells 51, 55, 60)
    regression_results = {
        "Linear Regression": 1.0, 
        "Random Forest Regressor": 0.9999937211277004, 
        "XGBoost Regressor": 0.9999937211277004, 
    }
    
    results_df = pd.DataFrame(regression_results.items(), columns=['Model', 'R² Score'])
    results_df['R² Score'] = results_df['R² Score'].map(lambda x: f"{x:.9f}")
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
