# =====================================================
# 🧠 Machine Learning Prediction App using Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ------------------------------
# 🎯 Page Title
# ------------------------------
st.set_page_config(page_title="Machine Learning Prediction App", layout="wide")
st.title("🔮 Machine Learning Prediction App")
st.write("Upload your CSV file, train a model, and make predictions interactively!")

# ------------------------------
# 📁 File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # ------------------------------
    # 🧭 Data Info
    # ------------------------------
    st.subheader("📋 Dataset Information")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    st.write("Columns:", list(data.columns))

    # ------------------------------
    # 🎯 Target Column Selection
    # ------------------------------
    target_column = st.selectbox("Select the target column (column to predict):", data.columns)

    if target_column:
        # Feature and Target Split
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle categorical features automatically
        X = pd.get_dummies(X)

        # ------------------------------
        # 🧩 Problem Type Selection
        # ------------------------------
        problem_type = st.radio("Select Problem Type:", ["Classification", "Regression"])

        # ------------------------------
        # ⚙️ Train-Test Split
        # ------------------------------
        test_size = st.slider("Select Test Size (Fraction):", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # ------------------------------
        # 🚀 Model Training
        # ------------------------------
        if st.button("Train Model"):
            if problem_type == "Classification":
                model = RandomForestClassifier(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Model training complete!")

            # ------------------------------
            # 📊 Evaluation Metrics
            # ------------------------------
            st.subheader("📈 Model Performance")
            if problem_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.metric(label="Accuracy", value=f"{acc:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.3f}")

            # ------------------------------
            # 🔍 Prediction Section
            # ------------------------------
            st.subheader("🔍 Make New Predictions")

            input_data = {}
            for col in X.columns:
                value = st.text_input(f"Enter value for {col}:", "")
                # Try to convert numeric columns
                try:
                    input_data[col] = float(value)
                except:
                    input_data[col] = value

            if st.button("Predict New Value"):
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)

                prediction = model.predict(input_df)[0]
                st.success(f"🎯 Predicted Value: {prediction}")

else:
    st.info("👆 Upload a CSV file to get started.")
