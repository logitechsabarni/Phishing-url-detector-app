import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔐", layout="wide")
st.title("🔐 AI-Based Phishing URL Detection System")

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
# TRAIN MODEL (Cached)
# ----------------------------
@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("dataset.csv")

    df["features"] = df["url"].apply(extract_features)

    X = np.array(df["features"].tolist())
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

rf, X_test, y_test = train_model()

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
        # PIE CHART
        # ----------------------------
        st.subheader("📊 Probability Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(
            [safe_prob, phishing_prob],
            labels=["Safe", "Phishing"],
            autopct="%1.1f%%",
            colors=["green", "red"],
            explode=(0.05, 0.05),
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # ----------------------------
        # FEATURE IMPORTANCE
        # ----------------------------
        st.subheader("📈 Model Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_,
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.set_xlabel("Importance Score")
        st.pyplot(fig2)

        # ----------------------------
        # SHAP EXPLANATION
        # ----------------------------
        st.subheader("🧠 SHAP Explanation")

        explainer = shap.Explainer(rf)
        shap_values = explainer(features_array)

        shap_vals = shap_values.values

        # Handle binary classifier shape safely
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0]
            if shap_vals.shape[0] == 2:
                shap_vals = shap_vals[1]
            else:
                shap_vals = shap_vals[0]
        elif shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        shap_vals = shap_vals.flatten()

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_vals,
        })

        shap_df["AbsImpact"] = np.abs(shap_df["Impact"])
        shap_df = shap_df.sort_values(by="AbsImpact", ascending=True)

        st.dataframe(
            shap_df[["Feature", "Impact"]]
            .style.format({"Impact": "{:.6f}"})
        )

        fig3, ax3 = plt.subplots()
        ax3.barh(shap_df["Feature"], shap_df["Impact"])
        ax3.set_xlabel("SHAP Impact")
        ax3.set_title("Feature Contribution to This Prediction")
        st.pyplot(fig3)

        # ----------------------------
        # DYNAMIC CONFUSION MATRIX
        # ----------------------------
        st.subheader("📊 Model Confusion Matrix")

        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig4, ax4 = plt.subplots()
        ax4.imshow(cm, cmap="Blues")

        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Safe", "Phishing"])
        ax4.set_yticklabels(["Safe", "Phishing"])

        for i in range(2):
            for j in range(2):
                ax4.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        ax4.set_title("Confusion Matrix")

        st.pyplot(fig4)

        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(e)
