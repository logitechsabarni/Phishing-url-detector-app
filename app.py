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
# FEATURE EXTRACTION (⚠ MUST MATCH TRAINING)
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

        safe_prob = float(probabilities[0])
        phishing_prob = float(probabilities[1])

        # 🔥 Proper Confidence Formula
        confidence = abs(phishing_prob - safe_prob) * 100

        st.subheader("🔎 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Phishing Website Detected")
        else:
            st.success("✅ Safe Website")

        # ----------------------------
        # PROBABILITY DISPLAY
        # ----------------------------
        col1, col2 = st.columns(2)

        col1.metric("Safe Probability", f"{safe_prob*100:.2f}%")
        col2.metric("Phishing Probability", f"{phishing_prob*100:.2f}%")

        st.metric("Model Confidence Level", f"{confidence:.2f}%")

        st.markdown("---")

        # ----------------------------
        # PIE CHART
        # ----------------------------
        st.subheader("📊 Probability Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(
            [safe_prob, phishing_prob],
            labels=["Safe", "Phishing"],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#FF4B4B"],
            explode=(0.05, 0.05),
            textprops={'fontsize': 12}
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        st.markdown("---")

        # ----------------------------
        # FEATURE IMPORTANCE
        # ----------------------------
        st.subheader("📈 Model Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Random Forest Feature Importance")
        st.pyplot(fig2)

        st.markdown("---")

        # ----------------------------
        # SHAP EXPLANATION
        # ----------------------------
        st.subheader("🧠 SHAP Explanation (Why this prediction?)")

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features_array)

        # Handle different SHAP versions safely
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact on Prediction": shap_vals
        }).sort_values(by="Impact on Prediction", key=abs, ascending=True)

        st.dataframe(shap_df.style.format({"Impact on Prediction": "{:.5f}"}))

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(shap_df["Feature"], shap_df["Impact on Prediction"])
        ax3.set_title("SHAP Feature Impact")
        ax3.set_xlabel("Impact Value")
        st.pyplot(fig3)

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(e)

# ----------------------------
# CONFUSION MATRIX (Demo)
# ----------------------------
st.markdown("---")
st.subheader("📊 Model Confusion Matrix (Evaluation Demo)")

# Replace with real test data if available
y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)

fig4, ax4 = plt.subplots()
ax4.imshow(cm, cmap="Blues")

ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Safe", "Phishing"])
ax4.set_yticklabels(["Safe", "Phishing"])

for i in range(2):
    for j in range(2):
        ax4.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)

ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title("Confusion Matrix")

st.pyplot(fig4)
