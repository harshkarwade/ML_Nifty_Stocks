import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from time import sleep

# ------------------------------
# 🎯 Page Setup
# ------------------------------
st.set_page_config(page_title="📈 ML Regression Dashboard", layout="wide")
st.title("🤖 Machine Learning Regression App")
st.markdown(
    """
    Upload your dataset, train a regression model, and make accurate predictions — interactively!  
    *(Currently supports Random Forest Regression.)*
    """
)

# ------------------------------
# 📁 File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    # ------------------------------
    # 📊 Dataset Preview
    # ------------------------------
    with st.expander("🔍 View Dataset Preview"):
        st.dataframe(data.head())

    # ------------------------------
    # 📋 Dataset Information
    # ------------------------------
    with st.expander("📘 Dataset Info"):
        st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
        st.write("**Columns:**", list(data.columns))

    # ------------------------------
    # 🎯 Target Column
    # ------------------------------
    target_column = st.selectbox("🎯 Select the Target (Value to Predict):", data.columns)

    if target_column:
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle categorical features
        X = pd.get_dummies(X)

        # ------------------------------
        # ⚙️ Model Parameters
        # ------------------------------
        st.subheader("⚙️ Model Configuration")
        test_size = st.slider("📏 Test Data Size (Fraction)", 0.1, 0.5, 0.2)
        n_estimators = st.slider("🌲 Number of Trees (Estimators)", 50, 300, 100, step=10)
        max_depth = st.slider("🌀 Maximum Depth of Trees", 2, 20, 8)

        # ------------------------------
        # 🚀 Train the Model
        # ------------------------------
        if st.button("🚀 Train Regression Model"):
            with st.spinner("🧠 Training your Random Forest Regressor..."):
                sleep(2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            st.success("🎉 Model Training Complete!")

            # ------------------------------
            # 📈 Model Performance Metrics
            # ------------------------------
            st.subheader("📊 Model Performance Metrics")

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("📉 RMSE", f"{rmse:.3f}")
            col2.metric("📏 MAE", f"{mae:.3f}")
            col3.metric("📈 R² Score", f"{r2:.3f}")

            # ------------------------------
            # 📊 Visual Comparison Chart
            # ------------------------------
            st.subheader("📊 Predicted vs Actual Values")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # ------------------------------
            # 🔍 Make New Predictions
            # ------------------------------
            st.subheader("🔮 Make a New Prediction")

            input_data = {}
            cols = st.columns(2)
            for i, col in enumerate(X.columns):
                with cols[i % 2]:
                    val = st.text_input(f"Enter {col}:")
                    try:
                        input_data[col] = float(val)
                    except:
                        input_data[col] = val

            if st.button("🎯 Predict Value"):
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)

                pred_value = model.predict(input_df)[0]
                st.success(f"🌟 Predicted Value: **{pred_value:.3f}**")

else:
    st.info("👆 Upload a CSV file to get started!")
