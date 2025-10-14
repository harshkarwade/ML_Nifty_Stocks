import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# --- Functions from the Jupyter Notebook ---

# Replicating the RSI function for completeness, though it's mainly for feature creation
def calculate_rsi(data, window=50):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Using ewm with com=window-1 for standard RSI calculation
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Placeholder function to simulate loading and preprocessing the data
@st.cache_data
def load_and_preprocess_data():
    """
    Simulates loading and preprocessing the data based on the notebook.
    In a real scenario, you would replace this with:
    df = pd.read_csv("Nifty_Stocks.csv")
    """
    # Create an empty DataFrame with the expected columns and types
    # Since the full data isn't available, we'll create dummy structured data
    st.info("Simulating data loading and preprocessing. Please replace this with your actual `pd.read_csv` and preprocessing logic if running locally.")
    
    # Load the actual Nifty Stocks data if available, or simulate (as done below)
    # We will assume a small subset for demonstration
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                 '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Open': [3790.0, 3811.1, 3767.0, 3701.75, 3675.0, 100.0, 101.0, 98.0, 99.0, 102.0],
        'High': [3832.0, 3811.1, 3771.85, 3719.0, 3747.75, 103.0, 103.0, 99.0, 101.0, 104.0],
        'Low': [3773.0, 3767.25, 3687.05, 3651.0, 3674.85, 99.0, 97.0, 97.0, 98.0, 101.0],
        'Close': [3811.1, 3783.2, 3691.75, 3666.8, 3737.9, 101.5, 97.5, 98.5, 100.5, 103.5],
        'Volume': [825907, 1344068, 1803075, 3598144, 1963127, 500000, 550000, 600000, 650000, 700000],
        'Symbol': ['TCS', 'TCS', 'TCS', 'TCS', 'TCS', 'INFY', 'INFY', 'INFY', 'INFY', 'INFY'],
        'Category': ['IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry', 'IT_industry'],
    }
    # Create the DataFrame
    df = pd.DataFrame(data)

    # 1. Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # 2. Re-create SMA_50, SMA_200, and RSI for visualization purposes
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean().fillna(0))
    df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean().fillna(0))
    # Note: RSI calculation in notebook uses 'window=50' which is unconventional but is replicated
    df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x, window=50).fillna(0))
    
    # 3. Drop rows that have any missing values introduced by rolling window (if using full data)
    # With the dummy data, everything will have values after fillna(0)

    # 4. For plotting the pattern, we only need a few columns.
    
    return df

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide")
    st.title("Stock Price Pattern Visualizer and Regression Analysis")

    # Load data
    df = load_and_preprocess_data()
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Filter Data")

    # 1. Select Symbol
    symbols = df['Symbol'].unique()
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)

    # Filter data by selected symbol
    df_symbol = df[df['Symbol'] == selected_symbol].sort_values(by='Date')

    if df_symbol.empty:
        st.error(f"No data available for symbol: {selected_symbol}")
        return

    # 2. Select Date Range
    min_date = df_symbol['Date'].min().to_pydatetime()
    max_date = df_symbol['Date'].max().to_pydatetime()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        # Filter by date range
        df_filtered = df_symbol[(df_symbol['Date'] >= pd.to_datetime(start_date)) & 
                                (df_symbol['Date'] <= pd.to_datetime(end_date))]
    else:
        # If only one date is selected, show data up to that date (or handle as an error)
        df_filtered = df_symbol
        st.warning("Please select a start and end date.")

    # --- Main Content ---
    
    st.header(f"Price Pattern for {selected_symbol}")
    
    # Check if filtered data is empty
    if df_filtered.empty:
        st.warning("No data found for the selected date range.")
    else:
        # --- 1. Price and Moving Averages Plot (Pattern Visualisation) ---
        
        # Create figure with secondary y-axis for Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.7, 0.3], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

        # Row 1: Price and SMAs
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], 
                                mode='lines', name='Close Price', line=dict(color='blue')), 
                      row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_50'], 
                                mode='lines', name='SMA 50', line=dict(color='orange', dash='dash')), 
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_200'], 
                                mode='lines', name='SMA 200', line=dict(color='red', dash='dot')), 
                      row=1, col=1)

        # Row 2: Volume
        fig.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['Volume'], 
                            name='Volume', marker_color='grey'), 
                      row=2, col=1)
        
        # Update layout and titles
        fig.update_layout(height=600, 
                          title_text=f"Stock Price Movement and Moving Averages for {selected_symbol}",
                          xaxis2_title="Date",
                          yaxis1_title="Price (Close, SMA)",
                          yaxis2_title="Volume",
                          hovermode="x unified",
                          margin=dict(l=20, r=20, t=50, b=20))

        st.plotly_chart(fig, use_container_width=True)

        # --- 2. Technical Indicator: RSI Plot ---
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure(data=[
            go.Scatter(x=df_filtered['Date'], y=df_filtered['RSI'], 
                       mode='lines', name='RSI', line=dict(color='purple'))
        ])
        # Add Overbought (70) and Oversold (30) lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(height=300, 
                              yaxis_title="RSI (50-Day)",
                              xaxis_title="Date",
                              margin=dict(l=20, r=20, t=20, b=20))
        
        st.plotly_chart(fig_rsi, use_container_width=True)

    st.markdown("---")
    
    # --- 3. Regression Model Summary ---
    st.header("Regression Model Performance")
    st.write("The Jupyter notebook trained three regression models to predict the 'Close' price:")

    # Results from the notebook (Cells 51, 55, 60)
    regression_results = {
        "Linear Regression": 1.0,  # r2 in cell 51
        "Random Forest Regressor": 0.9999937211277004,  # r21 in cell 55
        "XGBoost Regressor": 0.9999937211277004,  # r211 in cell 60
    }
    
    results_df = pd.DataFrame(regression_results.items(), columns=['Model', 'R² Score'])
    results_df['R² Score'] = results_df['R² Score'].map(lambda x: f"{x:.6f}")
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.success("The high R² scores suggest that the models are nearly perfectly predicting the 'Close' price, which is common in time series models where future price is heavily dependent on current and prior prices (e.g., Open, High, Low, and engineered features like SMA, MACD).")

if __name__ == "__main__":
    main()
