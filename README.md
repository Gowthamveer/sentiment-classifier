# 🔄 Customer Churn Prediction MLOps Platform

> An end-to-end production-grade MLOps pipeline that predicts customer churn using machine learning — from data ingestion to deployment with a live interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

---

## 🚀 Live Demo
👉 [Click here to try the app](#) ← add your deployment link here

---

## 🎯 Problem Statement
Customer churn costs businesses billions every year. This platform helps companies identify customers who are likely to leave before they do — enabling proactive retention strategies that save revenue.

---

## 📊 Model Performance

| Model | ROC-AUC | F1 Score | Accuracy |
|-------|---------|----------|----------|
| 🏆 LightGBM | **0.8377** | **0.5976** | **78%** |
| RandomForest | 0.8373 | 0.6090 | 78.1% |
| XGBoost | 0.8371 | 0.5930 | 77.9% |

---

## 🔍 Key Insights from SHAP Analysis
The model identified these as the top drivers of churn:
1. **Month-to-month contract** — highest churn risk, no long term commitment
2. **Electronic check payment** — less engaged customers
3. **Charges per service** — customers feeling they overpay tend to leave

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Class Imbalance | SMOTE (imbalanced-learn) |
| ML Models | LightGBM, XGBoost, RandomForest |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| REST API | FastAPI |
| Frontend Dashboard | Streamlit + Plotly |
| Version Control | Git + GitHub |

---

## 📁 Project Structure
```
churn-prediction-mlops/
├── data/
│   ├── raw/                    # Raw IBM Telco dataset
│   └── processed/              # Preprocessed & scaled data
├── src/
│   ├── preprocess.py           # Data cleaning & feature engineering
│   └── train.py                # Multi-model training with MLflow
├── api/
│   └── app.py                  # FastAPI REST API
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ How to Run Locally

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/Gowthamveer/churn-prediction-mlops.git
cd churn-prediction-mlops
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Preprocess data:**
```bash
python src/preprocess.py
```

**Step 4 — Train models:**
```bash
python src/train.py
```

**Step 5 — Run the dashboard:**
```bash
streamlit run frontend/app.py
```

**Step 6 — Run the API:**
```bash
uvicorn api.app:app --reload
```

---

## 🖥️ Dashboard Features

**📊 Dashboard Page**
- Total customers, churn rate, average monthly charges
- Churn by contract type, tenure distribution, internet service analysis

**🔍 Single Prediction Page**
- Enter customer details across 3 categories
- Get instant churn probability with risk level
- Interactive gauge chart showing confidence

**📁 Batch Prediction Page**
- Upload CSV of multiple customers
- Download results with risk levels

**📈 Model Performance Page**
- SHAP feature importance chart
- Side by side model comparison
- Full metrics breakdown

---

## 🧠 ML Pipeline
```
Raw Data → Data Cleaning → Feature Engineering → 
SMOTE Balancing → StandardScaler → 
Multi-Model Training → MLflow Tracking → 
Best Model Selection → FastAPI → Streamlit
```

---

## 📈 Business Impact
- Identifies **high risk customers** before they churn
- Provides **explainable predictions** so business teams understand why
- **Batch prediction** allows processing thousands of customers at once
- **Risk levels** (High/Medium/Low) make it actionable for non-technical teams

---

## 👨‍💻 Author
**Gowthamveer**
- GitHub: [@Gowthamveer](https://github.com/Gowthamveer)
- Live Demo: [Hugging Face Spaces](#)

---

## 📄 License
MIT License
