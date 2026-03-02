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

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Phishing URL Detection Dashboard",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Phishing URL Detection Dashboard")
st.markdown("Machine Learning based phishing URL classifier with full explainability")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("phishing_rf_model.pkl")

rf = load_model()

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
feature_names = [
    "URL Length",
    "Dot Count",
    "@ Symbol",
    "IP Address",
    "HTTPS",
    "Hyphen Count",
    "Slash Count",
    "Suspicious Keyword",
    "Domain Length"
]

def extract_features(url):

    features = []

    features.append(len(url))
    features.append(url.count("."))
    features.append(1 if "@" in url else 0)

    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    features.append(1 if re.search(ip_pattern, url) else 0)

    features.append(1 if url.startswith("https") else 0)
    features.append(url.count("-"))
    features.append(url.count("/"))

    suspicious_words = ["login", "verify", "secure",
                        "update", "account", "bank", "paypal"]

    features.append(
        1 if any(word in url.lower() for word in suspicious_words) else 0
    )

    parsed = urlparse(url)
    domain = parsed.netloc
    features.append(len(domain))

    return features


# --------------------------------------------------
# LOAD DATASET FOR METRICS
# --------------------------------------------------
df = pd.read_csv("dataset.csv")

X = np.array(df['url'].apply(extract_features).tolist())
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = rf.predict(X_test)

# --------------------------------------------------
# USER INPUT SECTION
# --------------------------------------------------
st.header("🔎 URL Prediction")

user_input = st.text_input("Enter URL to check:")

if st.button("Analyze URL") and user_input:

    features = extract_features(user_input)
    features_array = np.array(features).reshape(1, -1)

    prediction = rf.predict(features_array)[0]
    probability = rf.predict_proba(features_array)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Phishing URL Detected!")
        else:
            st.success("✅ Safe URL")

        st.metric("Phishing Probability", f"{round(probability*100,2)}%")

    # --------------------------------------------------
    # PIE CHART
    # --------------------------------------------------
    with col2:
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(
            [probability, 1 - probability],
            labels=["Phishing", "Safe"],
            autopct='%1.1f%%'
        )
        ax_pie.set_title("Prediction Distribution")
        st.pyplot(fig_pie)

    # --------------------------------------------------
    # SHAP FOR SINGLE URL (Readable)
    # --------------------------------------------------
    st.subheader("🧠 SHAP Explanation for This URL")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(features_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    fig_bar, ax_bar = plt.subplots()
    ax_bar.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax_bar.set_title("Feature Impact on This Prediction")
    st.pyplot(fig_bar)


# --------------------------------------------------
# MODEL PERFORMANCE SECTION
# --------------------------------------------------
st.header("📊 Model Performance Overview")

accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Safe", "Phishing"])
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# --------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------
st.subheader("📈 Global Feature Importance")

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig_imp, ax_imp = plt.subplots()
ax_imp.barh(importance_df["Feature"], importance_df["Importance"])
ax_imp.set_title("Random Forest Feature Importance")
st.pyplot(fig_imp)

# --------------------------------------------------
# GLOBAL SHAP SUMMARY
# --------------------------------------------------
st.subheader("🧠 Global SHAP Summary")

try:
    explainer = shap.TreeExplainer(rf)
    sample = X_test[:100]
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.clf()
    shap.summary_plot(shap_values, sample,
                      feature_names=feature_names,
                      show=False)
    st.pyplot(plt.gcf())

except:
    st.warning("SHAP summary not available.")
