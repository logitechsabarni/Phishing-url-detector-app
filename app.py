import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import shap
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Phishing URL Detection System")
st.markdown("Machine Learning based phishing URL classifier with explainability")

# -----------------------------------
# Load Model
# -----------------------------------
rf = joblib.load("phishing_rf_model.pkl")

# -----------------------------------
# Feature Extraction Function
# -----------------------------------
def extract_features(url):

    features = []

    features.append(len(url))                     # URL Length
    features.append(url.count("."))               # Dot count
    features.append(1 if "@" in url else 0)      # @ symbol

    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    features.append(1 if re.search(ip_pattern, url) else 0)  # IP present

    features.append(1 if url.startswith("https") else 0)      # HTTPS
    features.append(url.count("-"))                           # Hyphen count
    features.append(url.count("/"))                           # Slash count

    suspicious_words = ["login", "verify", "secure",
                        "update", "account", "bank", "paypal"]

    features.append(
        1 if any(word in url.lower() for word in suspicious_words) else 0
    )

    parsed = urlparse(url)
    domain = parsed.netloc
    features.append(len(domain))                   # Domain length

    return features


# -----------------------------------
# Prediction Section
# -----------------------------------
st.subheader("🔎 Test a URL")

user_input = st.text_input("Enter URL here:")

if st.button("Check URL"):

    features = extract_features(user_input)
    features = np.array(features).reshape(1, -1)

    prediction = rf.predict(features)[0]
    probability = rf.predict_proba(features)[0][1]

    if prediction == 1:
        st.error("⚠️ Phishing URL Detected!")
    else:
        st.success("✅ Safe URL")

    st.write("Confidence Score:", round(probability, 4))


# -----------------------------------
# Model Evaluation Section
# -----------------------------------
st.subheader("📊 Model Performance")

df = pd.read_csv("dataset.csv")

X = np.array(df['url'].apply(extract_features).tolist())
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write("Model Accuracy:", round(accuracy, 4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
st.pyplot(fig_cm)


# -----------------------------------
# SHAP Explainability
# -----------------------------------
st.subheader("🧠 SHAP Feature Importance")

explainer = shap.TreeExplainer(rf)
sample = X_test[:200]
shap_values = explainer.shap_values(sample)

fig_shap = plt.figure()
shap.summary_plot(shap_values[1], sample, show=False)
st.pyplot(fig_shap)
