import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from time import sleep

# ------------------------------
# 🎯 Page Setup
# ------------------------------
st.set_page_config(page_title="💡 Smart ML Prediction App", layout="wide")
st.title("🤖 Smart Machine Learning Prediction App")
st.markdown(
    """
    Upload your dataset, train a model, and make AI-powered predictions — all in one place!  
    *(Supports both Classification and Regression problems.)*
    """
)

# ------------------------------
# 📁 File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file to begin", type=["csv"])

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    # ------------------------------
    # 📊 Dataset Preview
    # ------------------------------
    with st.expander("🔍 Click to View Dataset Preview"):
        st.dataframe(data.head(10))

    # ------------------------------
    # 📋 Dataset Information
    # ------------------------------
    with st.expander("📘 Dataset Details"):
        st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
        st.write("**Columns:**", list(data.columns))

    # ------------------------------
    # 🎯 Target Column Selection
    # ------------------------------
    target_column = st.selectbox("🎯 Select the Target Column:", data.columns)

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle categorical variables automatically
        X = pd.get_dummies(X)

        # ------------------------------
        # 🧩 Choose Problem Type
        # ------------------------------
        st.subheader("⚙️ Model Configuration")
        problem_type = st.radio("What type of problem is this?", ["Classification", "Regression"], horizontal=True)
        test_size = st.slider("📏 Test Data Size (Fraction)", 0.1, 0.5, 0.2)

        # ------------------------------
        # 🚀 Model Training
        # ------------------------------
        if st.button("🚀 Train Model"):
            with st.spinner("🧠 Training your model... Please wait."):
                sleep(2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                if problem_type == "Classification":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.success("🎉 Model Training Complete!")

                # ------------------------------
                # 📈 Model Performance
                # ------------------------------
                st.subheader("📊 Model Performance Metrics")
                col1, col2 = st.columns(2)
                if problem_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    col1.metric("✅ Accuracy", f"{acc*100:.2f}%")
                    col2.metric("📉 Error Rate", f"{(1-acc)*100:.2f}%")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    col1.metric("📉 RMSE", f"{rmse:.3f}")
                    col2.metric("📏 MSE", f"{mse:.3f}")

                st.balloons()

                # ------------------------------
                # 🔍 Make New Predictions
                # ------------------------------
                st.subheader("🔮 Try New Predictions")
                st.write("Enter feature values to predict a new outcome:")

                input_data = {}
                cols = st.columns(2)
                for i, col in enumerate(X.columns):
                    with cols[i % 2]:
                        value = st.text_input(f"🔸 {col}", "")
                        try:
                            input_data[col] = float(value)
                        except:
                            input_data[col] = value

                if st.button("🎯 Predict Now"):
                    input_df = pd.DataFrame([input_data])
                    input_df = pd.get_dummies(input_df)
                    input_df = input_df.reindex(columns=X.columns, fill_value=0)

                    prediction = model.predict(input_df)[0]

                    st.success(f"🌟 Predicted Value: **{prediction}**")

else:
    st.info("👆 Upload a CSV file above to start your ML journey!")
