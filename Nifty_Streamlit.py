import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# --- Technical Indicator Functions from the Notebook ---

def calculate_rsi(data, window=50):
    """Calculates the Relative Strength Index (RSI)."""
    # 1. Calculate price changes
    delta = data.diff(1)
    # 2. Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 3. Calculate the smoothed average of gains and losses (using span=window for typical EWM)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    # 4. Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    # 5. Calculate the RSI
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
        # Load the raw file from the path used in your notebook (Cell 3)
        # NOTE: You MUST replace this with the correct path or ensure the file 
        # is available in your Streamlit Cloud environment.
        df = pd.read_csv("Nifty_Stocks.csv") 
    except FileNotFoundError:
        st.error("🚨 Error: 'Nifty_Stocks.csv' not found. Please ensure the file is in the correct directory or adjust the `pd.read_csv` path.")
        st.stop()

    # Data Cleaning and Feature Engineering (from Cells 7, 9, 12, 15, 16, 22, 23)
    
    # Fill NaN from original file (Cell 9)
    df = df.fillna(0)
    
    # Convert Date to datetime (Cell 12)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate SMA_50 and SMA_200 (Cells 15 & 16) - done per symbol in practice
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
    df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean())
    
    # Calculate RSI (Cells 21 & 22) - done per symbol
    df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x, window=50))
    
    # Fill NaNs created by rolling windows (Cells 19 & 23)
    df['SMA_50'] = df['SMA_50'].fillna(0)
    df['SMA_200'] = df['SMA_200'].fillna(0)
    df['RSI'] = df['RSI'].fillna(0)
    
    # Drop columns that are only needed for regression training and not plotting
    df = df.drop(columns=['Adj Close', 'Daily_Return', 'Price_Range', 'Volatility', 
                          'Cumulative_Return', 'Average_Price', 
                          'MACD12', 'MACD26', 'MACD9', 
                          'RollingVolatility', 'Daily_volatility', 'Annual_volatility'], 
                          errors='ignore')
    
    return df

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Stock Price Pattern Visualizer")
    st.title("📈 Stock Price Pattern Visualizer")
    st.markdown("Select a stock and date range to analyze price patterns and indicators.")
    st.markdown("---")

    # Load data
    df = load_and_preprocess_data()
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Filter & Analysis Options")

    # 1. Select Symbol
    symbols = sorted(df['Symbol'].unique())
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols, index=0)

    # Filter data by selected symbol
    df_symbol = df[df['Symbol'] == selected_symbol].sort_values(by='Date').reset_index(drop=True)

    if df_symbol.empty:
        st.error(f"No data available for symbol: **{selected_symbol}**")
        return

    # 2. Select Date Range
    min_date = df_symbol['Date'].min().to_pydatetime().date()
    max_date = df_symbol['Date'].max().to_pydatetime().date()

    # Set default range to the last 90 trading days or the whole period if shorter
    default_start_date = max_date - pd.DateOffset(days=120) 
    default_start_date = max(default_start_date.date(), min_date) if default_start_date.date() < max_date else min_date
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_symbol[(df_symbol['Date'] >= pd.to_datetime(start_date)) & 
                                (df_symbol['Date'] <= pd.to_datetime(end_date))]
    else:
        st.warning("Please select a valid start and end date for the visualization.")
        df_filtered = pd.DataFrame() # Empty dataframe if date selection is incomplete

    # --- Main Content: Visualization ---
    
    if not df_filtered.empty:
        # --- 1. Price and Moving Averages Plot (Pattern Visualisation) ---
        st.subheader(f"Price Pattern and Technical Indicators for **{selected_symbol}**")
        
        # Create figure with 2 rows for Price/SMA and RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.7, 0.3], 
                            subplot_titles=[f"Close Price & Moving Averages", "Relative Strength Index (RSI)"])

        # Row 1: Price and SMAs
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], 
                                mode='lines', name='Close Price', line=dict(color='blue', width=2)), 
                      row=1, col=1)
        
        # Only show SMAs if they have non-zero values (i.e., enough data points)
        if (df_filtered['SMA_50'] != 0).any():
            fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_50'], 
                                    mode='lines', name='SMA 50', line=dict(color='orange', dash='dash')), 
                          row=1, col=1)

        if (df_filtered['SMA_200'] != 0).any():
            fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_200'], 
                                    mode='lines', name='SMA 200', line=dict(color='red', dash='dot')), 
                          row=1, col=1)
        
        fig.update_yaxes(title_text="Price (INR)", row=1, col=1)

        # Row 2: RSI
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['RSI'], 
                            mode='lines', name='RSI', line=dict(color='purple')), 
                      row=2, col=1)
        
        # Add Overbought (70) and Oversold (30) lines for context
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        
        fig.update_yaxes(title_text="RSI (50-Day)", range=[0, 100], row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Final layout adjustments
        fig.update_layout(height=700, 
                          hovermode="x unified",
                          margin=dict(l=20, r=20, t=50, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- 2. Regression Model Summary ---
    st.header("Regression Model Performance Summary")
    st.write("The notebook used three models to predict the stock's 'Close' price across the entire dataset. All models showed extremely high accuracy (R² score close to 1.0), which is typical when predicting the immediate next price using current and historical data like Close, Open, High, Low, and technical indicators.")

    # Results from the notebook (Cells 51, 55, 60)
    regression_results = {
        "Linear Regression": 1.0, 
        "Random Forest Regressor": 0.9999937211277004, 
        "XGBoost Regressor": 0.9999937211277004, 
    }
    
    results_df = pd.DataFrame(regression_results.items(), columns=['Model', 'R² Score'])
    results_df['R² Score'] = results_df['R² Score'].map(lambda x: f"{x:.9f}") # Displaying high precision
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
