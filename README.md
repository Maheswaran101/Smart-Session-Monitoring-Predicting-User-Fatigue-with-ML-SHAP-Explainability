# Python code to generate README.md for Decision Fatigue Project

readme_content = """
# Decision Fatigue Prediction Project

## 1. Project Overview

Decision fatigue affects cognitive performance during repetitive tasks. This project aims to predict fatigue levels (High, Medium, Low) based on simulated user session behavior.

**Goal:** Build an interpretable ML pipeline that identifies which features contribute most to fatigue, enabling better productivity monitoring and intervention strategies.

---

## 2. Dataset

Since real-world data was unavailable, we simulated session data with features inspired by cognitive workload research:

| Feature | Description |
|---------|-------------|
| user_id | Unique user identifier |
| session_id | Session number per user |
| session_duration | Duration of the session (minutes) |
| decision_count | Number of decisions made in session |
| undo_count | Number of actions undone |
| error_rate | Proportion of incorrect actions |
| break_taken | Binary flag if a break was taken |
| time_of_day | Morning, Afternoon, Evening |
| fatigue_score | Continuous fatigue score |
| fatigue_level | Target variable: High, Medium, Low |

---

## 3. Exploratory Data Analysis (EDA)

- Fatigue distribution: Medium most common, Low least common.  
- Key correlations:
  - Higher error_rate → higher fatigue  
  - Longer session_duration → higher fatigue  
  - Breaks reduce fatigue impact  

Visualizations include count plots, heatmaps, and feature distributions.

---

## 4. Data Preprocessing

- **Encoding:**  
  - Target fatigue_level → numeric (High=0, Low=1, Medium=2)  
  - Categorical time_of_day → one-hot encoding
- **Feature scaling:** StandardScaler applied for Logistic Regression
- **Train/test split:** 80/20

---

## 5. Model Pipeline

### 5.1 Baseline Model: Logistic Regression

- Weighted F1-score: 0.977  
- Advantages: interpretable, strong performance on structured data  

### 5.2 Advanced Models

| Model | Weighted F1-score |
|-------|-----------------|
| Random Forest | 0.922 |
| XGBoost | 0.934 |

- **Decision:** Logistic Regression selected as final model due to highest F1-score and inherent interpretability.

---

## 6. Model Explainability with SHAP

Although Logistic Regression is already interpretable, SHAP was applied to validate and unify explanations across models.

### 6.1 Global Insights

- Most influential features: error_rate, decision_count, session_duration  
- Protective feature: break_taken reduces fatigue  

### 6.2 Local Explanation

- SHAP force plots show how individual session behaviors affect fatigue prediction.
- Example: High error_rate + long session → High fatigue; breaks reduce fatigue.

### 6.3 Visualizations

- Global Feature Importance Bar Plot  
- SHAP Summary Plot  
- Force Plot for a single prediction  

---

## 7. Business Insights

- High error_rate sessions indicate cognitive overload.  
- Encouraging breaks and monitoring session duration can reduce fatigue.  
- Model can guide adaptive interfaces or alert users in real-time.

---

## 8. Deployment (Optional)

- Save model and scaler with joblib:

```python
import joblib
joblib.dump(lr, "fatigue_model.pkl")
joblib.dump(scaler, "scaler.pkl")
