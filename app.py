import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="FatigueInsight",
    layout="centered"
)

# ===============================
# Load Trained Artifacts
# ===============================
model = joblib.load("models/fatigue_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ===============================
# App Title & Description
# ===============================
st.title("FatigueInsight: Cognitive Fatigue Predictor")
st.write(
    "This application predicts cognitive fatigue levels "
    "based on user interaction and session behavior."
)

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Session Inputs")

session_duration = st.sidebar.number_input(
    "Session Duration (minutes)",
    min_value=1,
    value=30
)

decision_count = st.sidebar.number_input(
    "Decision Count",
    min_value=0,
    value=10
)

undo_count = st.sidebar.number_input(
    "Undo Count",
    min_value=0,
    value=2
)

error_rate = st.sidebar.number_input(
    "Error Rate (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

break_taken = st.sidebar.selectbox(
    "Break Taken?",
    ["No", "Yes"]
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening"]
)

# ===============================
# Prepare Input Data
# ===============================
input_df = pd.DataFrame({
    "session_duration": [session_duration],
    "decision_count": [decision_count],
    "undo_count": [undo_count],
    "error_rate": [error_rate],
    "break_taken": [1 if break_taken == "Yes" else 0],
    "time_of_day_Morning": [1 if time_of_day == "Morning" else 0],
    "time_of_day_Afternoon": [1 if time_of_day == "Afternoon" else 0],
    "time_of_day_Evening": [1 if time_of_day == "Evening" else 0],
})

# Align with training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ===============================
# Prediction
# ===============================
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

st.subheader("Prediction Result")

if prediction == 1:
    st.error("High Cognitive Fatigue Detected")
else:
    st.success("Low Cognitive Fatigue Detected")

# ===============================
# Feature Importance
# ===============================
st.subheader("Feature Importance")

importance = pd.Series(
    model.coef_[0],
    index=feature_names
).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
importance.plot(kind="barh", ax=ax)
ax.set_xlabel("Impact on Fatigue Prediction")
st.pyplot(fig)

# ===============================
# Explainability Note
# ===============================
st.info(
    "SHAP explainability is excluded in deployment due to "
    "environment constraints. Model coefficients are used "
    "for transparent interpretation."
)
