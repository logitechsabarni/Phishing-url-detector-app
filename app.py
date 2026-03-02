import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix

# ----------------------------
# LOAD MODEL
# ----------------------------
model = pickle.load(open("phishing_rf_model.pkl", "rb"))

# ----------------------------
# FEATURE EXTRACTION (9 FEATURES)
# ----------------------------
def extract_features(url):
    features = []
    
    # 1. URL Length
    features.append(len(url))
    
    # 2. Has @ symbol
    features.append(1 if "@" in url else 0)
    
    # 3. Number of dots
    features.append(url.count("."))
    
    # 4. Contains IP address
    ip_pattern = r"\d+\.\d+\.\d+\.\d+"
    features.append(1 if re.search(ip_pattern, url) else 0)
    
    # 5. Suspicious words
    suspicious_words = ["login", "verify", "update", "secure", "bank", "free", "bonus"]
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)
    
    # 6. Hyphen count
    features.append(url.count("-"))
    
    # 7. Question mark count
    features.append(url.count("?"))
    
    # 8. Equal sign count
    features.append(url.count("="))
    
    # 9. HTTPS present
    features.append(1 if "https" in url else 0)
    
    return np.array(features).reshape(1, -1)

feature_names = [
    "URL Length",
    "@ Present",
    "Dot Count",
    "IP Present",
    "Suspicious Words",
    "Hyphen Count",
    "? Count",
    "= Count",
    "HTTPS Present",
]

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Phishing URL Detector", page_icon="🔐")
st.title("🔐 Phishing URL Detector")
st.write("Enter a URL below to check whether it is Safe or Phishing.")

url = st.text_input("Enter URL")

if st.button("Check URL"):
    
    if url.strip() == "":
        st.warning("Please enter a URL")
    else:
        features = extract_features(url)
        prediction = model.predict(features)
        
        # ----------------------------
        # PROBABILITIES
        # ----------------------------
        try:
            probabilities = model.predict_proba(features)[0]
            safe_prob = float(probabilities[0])
            phishing_prob = float(probabilities[1])
        except:
            # If predict_proba not available, fall back
            safe_prob = 0.5
            phishing_prob = 0.5
        
        # ----------------------------
        # SHOW PREDICTION
        # ----------------------------
        if prediction[0] == 1:
            st.error("⚠️ Phishing Website Detected!")
        else:
            st.success("✅ This Website Looks Safe")
        
        col1, col2 = st.columns(2)
        col1.metric("Safe Probability", f"{safe_prob*100:.2f}%")
        col2.metric("Phishing Probability", f"{phishing_prob*100:.2f}%")
        
        # ----------------------------
        # FIGURE 1: PIE CHART
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
        # FIGURE 2: FEATURE IMPORTANCE
        # ----------------------------
        st.subheader("📈 Model Feature Importance")
        try:
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig2, ax2 = plt.subplots()
            ax2.barh(importance_df["Feature"], importance_df["Importance"])
            ax2.set_xlabel("Importance Score")
            ax2.set_title("Global Feature Importance")
            st.pyplot(fig2)
        except:
            st.info("Model feature importances not available")
        
        # ----------------------------
        # FIGURE 3: SHAP EXPLANATION
        # ----------------------------
        st.subheader("🧠 SHAP Explanation (Feature Contribution)")
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(features)
            
            if isinstance(shap_values, list):  # binary classifier
                shap_vals = shap_values[prediction[0]][0]
            else:
                shap_vals = shap_values[0]
                if shap_vals.size == 2*len(feature_names):
                    shap_vals = shap_vals[prediction[0]*len(feature_names):(prediction[0]+1)*len(feature_names)]
            
            shap_vals = shap_vals.flatten()
            
            if len(shap_vals) != len(feature_names):
                st.warning(f"SHAP explanation unavailable (expected {len(feature_names)} features, got {len(shap_vals)})")
            else:
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Impact on Prediction": shap_vals
                })
                shap_df["Absolute Impact"] = abs(shap_df["Impact on Prediction"])
                shap_df = shap_df.sort_values(by="Absolute Impact", ascending=True)
                
                st.dataframe(
                    shap_df[["Feature", "Impact on Prediction"]]
                    .style.format({"Impact on Prediction": "{:.6f}"})
                )
                
                fig3, ax3 = plt.subplots()
                ax3.barh(
                    shap_df["Feature"],
                    shap_df["Impact on Prediction"],
                    color=np.where(shap_df["Impact on Prediction"]>0, "#e74c3c", "#2ecc71")
                )
                ax3.set_xlabel("SHAP Value Impact")
                ax3.set_title("Feature Contribution for This URL")
                st.pyplot(fig3)
                
        except Exception as e:
            st.warning("SHAP explanation could not be generated")
            st.write(e)
        
        # ----------------------------
        # FIGURE 4: CONFUSION MATRIX
        # ----------------------------
        st.subheader("📊 Model Confusion Matrix (Demo)")
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
                ax4.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        ax4.set_title("Confusion Matrix")
        st.pyplot(fig4)
