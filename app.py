import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

scaler = joblib.load("models/scaler.pkl")
lr_model = joblib.load("models/logistic_regression.pkl")
autoencoder = load_model("models/autoencoder.h5", compile=False)

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

df = load_data()
features = df.columns.drop("Class")

X_scaled = scaler.transform(df[features].values)
X_nonfraud = X_scaled[df["Class"] == 0]
reconstructions = autoencoder.predict(X_nonfraud)
mse_nonfraud = np.mean(np.power(X_nonfraud - reconstructions, 2), axis=1)
threshold = np.mean(mse_nonfraud) + 3 * np.std(mse_nonfraud)

st.sidebar.title("⚙️ Options")
option = st.sidebar.radio("Choose an option:", ["🔍 Predict Transaction", "📊 Explore Dataset"])

# --- Top Dashboard Metrics ---
col1, col2, col3 = st.columns(3)
total_tx = len(df)
fraud_count = df["Class"].sum()
legit_count = total_tx - fraud_count
col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Legitimate", f"{legit_count:,}", delta=f"{(legit_count/total_tx)*100:.2f}%")
col3.metric("Fraudulent", f"{fraud_count:,}", delta=f"{(fraud_count/total_tx)*100:.2f}%")

if option == "📊 Explore Dataset":
    st.title("📊 Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### Class Distribution")
    st.bar_chart(df["Class"].value_counts())

if option == "🔍 Predict Transaction":
    st.title("🔍 Predict Transaction Fraud")

    sample_row = df.sample(1, random_state=np.random.randint(0, 10000))
    default_values = sample_row[features].iloc[0].to_dict()

    with st.form("transaction_form"):
        cols = st.columns(4)
        input_data = {}
        for i, feature in enumerate(features):
            with cols[i % 4]:
                input_data[feature] = st.number_input(feature, value=float(default_values[feature]))
        submitted = st.form_submit_button("Predict 🚀")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        lr_pred = lr_model.predict(input_scaled)[0]
        reconstruction = autoencoder.predict(input_scaled)
        mse = np.mean(np.power(input_scaled - reconstruction, 2))
        auto_pred = 1 if mse > threshold else 0
        final_pred = 1 if (lr_pred + auto_pred) >= 1 else 0

        st.subheader("🔎 Results")
        st.success(f"Logistic Regression: **{lr_pred}**")
        st.success(f"Autoencoder: **{auto_pred}** (threshold={threshold:.6f})")

        if final_pred == 1:
            st.error("🚨 Final Ensemble Prediction: FRAUD DETECTED")
        else:
            st.info("✅ Final Ensemble Prediction: Legitimate Transaction")
