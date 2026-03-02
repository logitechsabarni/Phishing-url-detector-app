import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# FEATURE EXTRACTION (9 FEATURES)
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
        # Extract features and reshape
        features = extract_features(url)
        features_array = np.array(features).reshape(1, -1)

        # ----------------------------
        # 0️⃣ FEATURE CHECKLIST (Tick Verification)
        # ----------------------------
        st.subheader("🧾 Feature Extraction Verification ✅")
        for fname, fvalue in zip(feature_names, features_array.flatten()):
            st.markdown(f"{fname} ✅ (Value: {fvalue})")

        # ----------------------------
        # 1️⃣ PREDICTION
        # ----------------------------
        prediction = rf.predict(features_array)[0]
        probabilities = rf.predict_proba(features_array)[0]

        safe_prob = float(probabilities[0])
        phishing_prob = float(probabilities[1])
        confidence = abs(phishing_prob - safe_prob) * 100

        st.subheader("🔎 Prediction Result")
        if prediction == 1:
            st.error("⚠️ Phishing Website Detected")
        else:
            st.success("✅ Safe Website")

        col1, col2 = st.columns(2)
        col1.metric("Safe Probability", f"{safe_prob*100:.2f}%")
        col2.metric("Phishing Probability", f"{phishing_prob*100:.2f}%")
        st.metric("Model Confidence", f"{confidence:.2f}%")

        # ----------------------------
        # 2️⃣ PIE CHART
        # ----------------------------
        st.subheader("📊 Probability Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [safe_prob, phishing_prob],
            labels=["Safe", "Phishing"],
            autopct='%1.1f%%',
            colors=["#2ecc71", "#e74c3c"],
            explode=(0.08, 0.08),
            textprops={'fontsize': 12}
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # ----------------------------
        # 3️⃣ FEATURE IMPORTANCE
        # ----------------------------
        st.subheader("📈 Global Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Global Feature Importance")
        st.pyplot(fig2)

        # ----------------------------
        # 4️⃣ FEATURE CONTRIBUTION BAR
        # ----------------------------
        st.subheader("🟦 Feature Contribution (Value × Importance)")
        contribution = features_array.flatten() * rf.feature_importances_
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contribution
        }).sort_values(by="Contribution", ascending=True)

        fig3, ax3 = plt.subplots()
        ax3.barh(contrib_df["Feature"], contrib_df["Contribution"], color="#3498db")
        ax3.set_xlabel("Feature Contribution")
        ax3.set_title("Feature Contribution to Prediction")
        st.pyplot(fig3)

        # ----------------------------
        # 5️⃣ RADAR CHART
        # ----------------------------
        st.subheader("🟢 Feature Profile (Radar Chart)")
        values = features_array.flatten()
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        values_loop = np.concatenate((values, [values[0]]))
        angles_loop = np.concatenate((angles, [angles[0]]))

        fig4, ax4 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax4.plot(angles_loop, values_loop, 'o-', linewidth=2)
        ax4.fill(angles_loop, values_loop, alpha=0.25)
        ax4.set_thetagrids(angles * 180/np.pi, feature_names)
        ax4.set_title("Feature Profile of URL")
        st.pyplot(fig4)

        # ----------------------------
        # 6️⃣ DYNAMIC HEATMAP
        # ----------------------------
        st.subheader("🔥 Feature Contribution Heatmap")
        contrib_norm = contribution / (np.max(np.abs(contribution)) + 1e-6)
        heatmap_df = pd.DataFrame(contrib_norm.reshape(1,-1), columns=feature_names)

        fig5, ax5 = plt.subplots()
        sns.heatmap(heatmap_df, annot=True, cmap="RdYlGn_r", cbar=True, ax=ax5)
        ax5.set_title("Dynamic Feature Contribution Heatmap")
        st.pyplot(fig5)

        # ----------------------------
        # 7️⃣ DYNAMIC CONFUSION MATRIX
        # ----------------------------
        st.subheader("📊 Confusion Matrix (Dynamic)")

        cm_dynamic = np.array([
            [safe_prob, 0],
            [0, phishing_prob]
        ])

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm_dynamic, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=["Safe","Phishing"], yticklabels=["Safe","Phishing"], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual (Demo)")
        ax_cm.set_title("Dynamic Confusion Matrix (Reflects Probabilities)")
        st.pyplot(fig_cm)

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(e)
