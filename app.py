import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔐", layout="wide")
st.title("🔐 AI-Based Phishing URL Detection System")

# ----------------------------
# LOAD MODEL
# ----------------------------
rf = joblib.load("phishing_rf_model.pkl")

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(url):
    return [
        len(url),
        url.count("-"),
        url.count("@"),
        url.count("?"),
        url.count("%"),
        url.count("."),
        url.count("="),
        url.count("http"),
        1 if "https" in url else 0,
        sum(c.isdigit() for c in url),
    ]

feature_names = [
    "URL Length",
    "Hyphen Count",
    "@ Count",
    "? Count",
    "% Count",
    "Dot Count",
    "= Count",
    "HTTP Count",
    "HTTPS Present",
    "Digit Count",
]

# ----------------------------
# USER INPUT
# ----------------------------
url = st.text_input("Enter URL to Analyze:")

if url:

    try:
        features = extract_features(url)
        features_array = np.array(features).reshape(1, -1)

        # ----------------------------
        # PREDICTION
        # ----------------------------
        prediction = rf.predict(features_array)[0]
        probabilities = rf.predict_proba(features_array)[0]

        phishing_prob = float(probabilities[1])
        safe_prob = float(probabilities[0])

        st.subheader("🔎 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Phishing Website Detected")
        else:
            st.success("✅ Safe Website")

        st.write(f"### Confidence Score: {max(phishing_prob, safe_prob)*100:.2f}%")

        # ----------------------------
        # PIE CHART
        # ----------------------------
        st.subheader("📊 Probability Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(
            [safe_prob, phishing_prob],
            labels=["Safe", "Phishing"],
            autopct='%1.1f%%',
            colors=["green", "red"],
            explode=(0.05, 0.05)
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # ----------------------------
        # FEATURE IMPORTANCE
        # ----------------------------
        st.subheader("📈 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance_df["Feature"], importance_df["Importance"], color="blue")
        st.pyplot(fig2)

        # ----------------------------
        # SHAP EXPLANATION
        # ----------------------------
        st.subheader("🧠 SHAP Explanation")

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features_array)

        # Robust SHAP handling
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        elif len(np.array(shap_values).shape) == 3:
            shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_vals
        }).sort_values(by="Impact", key=abs, ascending=True)

        st.dataframe(shap_df)

        fig3, ax3 = plt.subplots()
        ax3.barh(shap_df["Feature"], shap_df["Impact"], color="purple")
        st.pyplot(fig3)

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(e)

# ----------------------------
# CONFUSION MATRIX (STATIC DEMO)
# ----------------------------
st.subheader("📊 Model Confusion Matrix")

# Replace with real y_test if available
y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)

fig4, ax4 = plt.subplots()
ax4.imshow(cm, cmap="Blues")

ax4.set_xticks([0,1])
ax4.set_yticks([0,1])
ax4.set_xticklabels(["Safe","Phishing"])
ax4.set_yticklabels(["Safe","Phishing"])

for i in range(2):
    for j in range(2):
        ax4.text(j, i, cm[i, j], ha="center", va="center", color="black")

ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")

st.pyplot(fig4)
