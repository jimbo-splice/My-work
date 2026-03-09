# 💳 Credit Card Fraud Detection App

https://james-mcf-ccfraud.streamlit.app/

An interactive **machine learning web application** built with **Streamlit** that predicts whether a credit card transaction is fraudulent.  
The app allows users to manually input transactions, generate realistic random transactions, or upload a CSV file to perform batch fraud detection.

This project demonstrates **end-to-end ML workflow skills** including data preprocessing, model training, class imbalance handling, model evaluation, explainability, and deployment of a machine learning model in a web interface.

---

# 🚀 Features

### 🔍 Fraud Prediction
Users can predict fraud probability using three input methods:

**1. Manual Input**
- Enter transaction values manually
- Supports all 31 model features
- Displays prediction, fraud probability, and confidence score

**2. Random Transaction Generator**
- Generates realistic transactions using typical dataset percentiles
- Allows quick demonstration of model behaviour

**3. CSV Upload**
- Upload a CSV of transactions
- Batch prediction for large datasets
- Download predictions as a new CSV

---

### 📊 Model Information Page
Displays key details about the trained model:

- Model type
- Number of features
- Evaluation metric
- ROC curve visualisation

This helps demonstrate model performance and evaluation methodology.

---

### 📈 Feature Importance Page
Uses **SHAP (SHapley Additive Explanations)** to explain model predictions.

The SHAP summary plot shows:

- Which features influence fraud predictions most
- Whether features push predictions toward fraud or legitimate
- Relative magnitude of feature effects

Explainability is critical for financial ML systems.

---

# 🧠 Machine Learning Model

The model was trained on the widely used **European credit card fraud dataset**.

### Dataset Characteristics
- **284,807 transactions**
- **492 fraudulent transactions (0.17%)**
- Highly **imbalanced classification problem**

### Feature Structure
- `Time`
- `Amount`
- `Time_of_day`
- `V1 – V28`

The `V` features are **PCA-transformed variables** used to anonymize sensitive financial data.

---

### Handling Class Imbalance

Fraud detection datasets are extremely imbalanced.

The model training pipeline used:

- **SMOTE (Synthetic Minority Oversampling Technique)**  
to generate synthetic fraud samples.

This improves the model’s ability to detect rare fraudulent events.

---

### Model Evaluation

Evaluation focused on metrics suitable for imbalanced classification:

- **ROC-AUC**
- Precision
- Recall
- Fraud detection rate

The model achieved approximately:


