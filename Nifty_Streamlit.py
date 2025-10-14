import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# --- Technical Indicator Function from the Notebook ---

def calculate_rsi(data, window=50):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    # Adding a small epsilon to avoid division by zero in case of zero average loss
    rs = avg_gain / (avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Loads the uploaded CSV file and performs all necessary preprocessing 
    (Date conversion, SMA, RSI calculation, and filling NaNs).
    """
    # Use StringIO to read the content of the uploaded file
    try:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
    except Exception as e:
        st.error(f"Error reading CSV file. Please ensure it is a valid CSV format: {e}")
        return pd.DataFrame()

    # Data Cleaning and Feature Engineering (from the notebook logic)
    
    # 1. Fill NaN from original file
    df = df = df.fillna(0)
    
    # 2. Convert Date to datetime. Assuming 'Date' column is present.
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("The uploaded file must contain a 'Date' column.")
        return pd.DataFrame()
        
    # 3. Check for essential columns (Open, High, Low, Close, Symbol)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"The file must contain the following columns: {', '.join(missing_cols)} are missing.")
        return pd.DataFrame()

    # 4. Ensure 'Symbol' is treated as a string type for the multiselect later
    df['Symbol'] = df['Symbol'].astype(str)

    # 5. Calculate SMA_50, SMA_200, and RSI for each stock symbol
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean().fillna(0))
    df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean().fillna(0))
    df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x, window=50).fillna(0))
    
    return df.sort_values(by='Date')

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Stock Analysis & Comparison Tool")
    st.title("📊 Stock Data Analysis and Comparison Tool")
    st.markdown("Upload your Nifty Stock CSV file to begin analysis.")
    st.markdown("---")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_full = load_and_preprocess_data(uploaded_file)
        
        if df_full.empty:
            return

        st.success("File loaded and preprocessing complete! Use the sidebar filters to explore.")
        
        # --- Sidebar for User Input ---
        st.sidebar.header("Filter & Analysis Options")

        # 1. Select Multiple Symbols for Comparison
        all_symbols = sorted(df_full['Symbol'].unique())
        
        # Set a default selection: the first two stocks, or all if less than two
        default_selection = all_symbols[:2] if len(all_symbols) >= 2 else all_symbols

        selected_symbols = st.sidebar.multiselect(
            "Select Stock Symbols (for comparison)", 
            all_symbols, 
            default=default_selection
        )

        if not selected_symbols:
            st.warning("Please select at least one stock symbol to visualize.")
            return

        # Filter data for selected symbols
        df_filtered_symbols = df_full[df_full['Symbol'].isin(selected_symbols)].reset_index(drop=True)

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
            return

        # --- Main Content: Visualization ---
        
        if not df_final.empty:
            
            # 1. Price Comparison Plot
            # FIX APPLIED HERE: Use map(str, selected_symbols) to ensure all elements are strings
            st.subheader(f"Close Price Comparison: {', '.join(map(str, selected_symbols))}")
            
            fig = go.Figure()

            for symbol in selected_symbols:
                df_stock = df_final[df_final['Symbol'] == symbol]
                
                fig.add_trace(go.Scatter(
                    x=df_stock['Date'], 
                    y=df_stock['Close'], 
                    mode='lines', 
                    name=f'{symbol} Close Price'
                ))

            fig.update_layout(
                height=600, 
                title_text="Stock Price Pattern Comparison",
                xaxis_title="Date",
                yaxis_title="Close Price (INR)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)

            # 2. Detailed Technical Indicators (for the first selected stock)
            if selected_symbols:
                # Ensure the symbol is a string before using it in the subheader
                first_symbol = str(selected_symbols[0]) 
                st.subheader(f"Technical Indicators for Detailed Analysis: {first_symbol}")
                df_single = df_final[df_final['Symbol'] == first_symbol]
                
                fig_detail = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3], 
                                    subplot_titles=[f"{first_symbol} Price & Moving Averages", "Relative Strength Index (RSI)"])
                
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
                fig_detail.update_layout(height=500, showlegend=False)

                st.plotly_chart(fig_detail, use_container_width=True)

        st.markdown("---")
        
        # 3. Regression Model Summary
        st.header("Regression Model Performance Summary")
        st.write("This section reflects the R² scores achieved in the original notebook when predicting the 'Close' price across the entire dataset.")

        regression_results = {
            "Linear Regression": 1.0, 
            "Random Forest Regressor": 0.9999937211277004, 
            "XGBoost Regressor": 0.9999937211277004, 
        }
        
        results_df = pd.DataFrame(regression_results.items(), columns=['Model', 'R² Score'])
        results_df['R² Score'] = results_df['R² Score'].map(lambda x: f"{x:.9f}")
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    else:
        st.info("Please upload your Nifty Stock CSV file to activate the visualization and comparison features.")

if __name__ == "__main__":
    main()
