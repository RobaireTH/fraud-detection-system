import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#Load Model

@st.cache_data
def load_model():
    return joblib.load("../model/model_xgboost.pkl")

model = load_model()

# UI Title
st.title("CREDIT CARD FRAUD DETECTOR")

st.markdown("""
Upload a transaction dataset (CSV format) with the same structure as the Kaggle dataset. The app will predict fraudulent transactions and show evaluation metrics.   You should drop the time column. The app will require the target column <Class> for Evaluation Metrics, though you can still make predictions without it. 
  """)


uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Class" in df.columns:
        X = df.drop(columns=["Class"], axis=1)
        y = df["Class"]

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Show sample prediction
        df["Predicted"] = y_pred
        df["Probability"] = y_proba

        st.subheader("SAMPLE PREDICTIONS")
        st.dataframe(df[["Amount", "Predicted", "Probability"]].head(10))

        st.subheader("CLASSIFICATION REPORT")
        report = classification_report(y, y_pred, output_dict=True)
        auc = roc_auc_score(y, y_proba)
        f1 = f1_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        st.text(classification_report(y, y_pred))
        st.text(f"Precision: {precision: .4f}")
        st.text(f"Recall: {recall: .4f}")
        st.text(f"F1-Score: {f1: .4f}")
        st.text(f"AUC-ROC: {auc: .4f}")
        st.text(f"Accuracy: {accuracy_score(y, y_pred): .4f}")


        auc_score = roc_auc_score(y, y_proba)
        st.success(f"AUC-ROC Score: {auc_score: .4f}")

        st.subheader("CONFUSION MATRIX")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.download_button("Download predictions", df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.warning("Missing 'Class' column- this is required for evaluation. You can still predict by removing this check")
