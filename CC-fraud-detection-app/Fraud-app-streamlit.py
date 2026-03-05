
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_model():
    try:
        with open("fraud_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'fraud_model.pkl' is in this directory.")
        st.stop()

model = load_model()

# =============================================================================
# FEATURE ORDER (CRITICAL – must match training)
# =============================================================================

EXPECTED_FEATURES = (
    ['Time'] +
    [f'V{i}' for i in range(1, 29)] +
    ['Amount', 'Time_of_day']
)

# =============================================================================
# FEATURE RANGES (for random generation)
# =============================================================================

FEATURE_RANGES = {
    'Time': {'min': 54000, 'max': 130000, 'q25': 54000, 'q75': 130000},
    'Amount': {'min': 5.0, 'max': 77.0, 'q25': 5.65, 'q75': 77.17},
    'Time_of_day': {'min': 0, 'max': 23.99, 'q25': 8, 'q75': 18},
    'V1': {'min': -3, 'max': 2.5, 'q25': -0.92, 'q75': 1.32},
    'V2': {'min': -3, 'max': 3, 'q25': -0.60, 'q75': 0.80},
    'V3': {'min': -3, 'max': 2, 'q25': -0.89, 'q75': 1.03},
    'V4': {'min': -3, 'max': 3, 'q25': -0.85, 'q75': 0.74},
    'V5': {'min': -3, 'max': 2, 'q25': -0.69, 'q75': 0.61},
    'V6': {'min': -3, 'max': 2, 'q25': -0.77, 'q75': 0.40},
    'V7': {'min': -3, 'max': 2, 'q25': -0.55, 'q75': 0.57},
    'V8': {'min': -2, 'max': 2, 'q25': -0.21, 'q75': 0.33},
    'V9': {'min': -2, 'max': 2, 'q25': -0.64, 'q75': 0.60},
    'V10': {'min': -3, 'max': 2, 'q25': -0.54, 'q75': 0.45},
    'V11': {'min': -3, 'max': 2, 'q25': -0.76, 'q75': 0.74},
    'V12': {'min': -3, 'max': 3, 'q25': -0.41, 'q75': 0.62},
    'V13': {'min': -2, 'max': 2, 'q25': -0.65, 'q75': 0.66},
    'V14': {'min': -3, 'max': 2, 'q25': -0.43, 'q75': 0.49},
    'V15': {'min': -2, 'max': 2, 'q25': -0.58, 'q75': 0.65},
    'V16': {'min': -2, 'max': 2, 'q25': -0.47, 'q75': 0.52},
    'V17': {'min': -3, 'max': 2, 'q25': -0.48, 'q75': 0.40},
    'V18': {'min': -2, 'max': 2, 'q25': -0.50, 'q75': 0.50},
    'V19': {'min': -2, 'max': 2, 'q25': -0.46, 'q75': 0.46},
    'V20': {'min': -2, 'max': 2, 'q25': -0.21, 'q75': 0.13},
    'V21': {'min': -2, 'max': 2, 'q25': -0.23, 'q75': 0.19},
    'V22': {'min': -2, 'max': 2, 'q25': -0.54, 'q75': 0.53},
    'V23': {'min': -2, 'max': 2, 'q25': -0.16, 'q75': 0.15},
    'V24': {'min': -2, 'max': 2, 'q25': -0.35, 'q75': 0.44},
    'V25': {'min': -2, 'max': 2, 'q25': -0.32, 'q75': 0.35},
    'V26': {'min': -2, 'max': 2, 'q25': -0.33, 'q75': 0.24},
    'V27': {'min': -2, 'max': 2, 'q25': -0.07, 'q75': 0.09},
    'V28': {'min': -2, 'max': 2, 'q25': -0.05, 'q75': 0.08}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_random_transaction():
    """Generate random transaction data between 25th-75th percentile"""
    random_data = {}
    for feature, ranges in FEATURE_RANGES.items():
        random_data[feature] = np.random.uniform(ranges['q25'], ranges['q75'])
    return random_data

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Prediction", "Model Information", "Feature Importance"]
)

st.sidebar.divider()
st.sidebar.caption("Fraud Detection System | 2026")

# =============================================================================
# PREDICTION PAGE
# =============================================================================

if page == "Prediction":

    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("Predict whether a transaction is fraudulent.")

    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio(
        "Choose input type:",
        ["Manual Input", "Random Transaction", "Upload CSV"]
    )

# -------------------------------------------------------------------------
# MANUAL INPUT
# -------------------------------------------------------------------------

if input_method == "Manual Input":
    st.header("📝 Manual Transaction Input")
    
    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎲 Generate Random Values", use_container_width=True):
            st.session_state['random_data'] = generate_random_transaction()
    
    with col2:
        if st.button("🔄 Clear All Fields", use_container_width=True):
            if 'random_data' in st.session_state:
                del st.session_state['random_data']
            st.rerun()
    
    # Initialize session state
    if 'random_data' not in st.session_state:
        st.session_state['random_data'] = None
    
    # Create input form
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        # Time, Amount, and Time_of_day
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_val = st.session_state['random_data']['Time'] if st.session_state['random_data'] else 0.0
            time = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=float(time_val),
                step=100.0,
                format="%.2f",
                help="Seconds since first transaction"
            )
        
        with col2:
            amount_val = st.session_state['random_data']['Amount'] if st.session_state['random_data'] else 0.0
            amount = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=30000.0,
                value=float(amount_val),
                step=1.0,
                format="%.2f"
            )
        
        with col3:
            time_of_day_val = st.session_state['random_data']['Time_of_day'] if st.session_state['random_data'] else 0.0
            time_of_day = st.number_input(
                "Time of Day (hour)",
                min_value=0.0,
                max_value=23.99,
                value=float(time_of_day_val),
                step=0.1,
                format="%.2f",
                help="Hour of day (0-23)"
            )
        
        st.divider()
        st.subheader("PCA Features (V1-V28)")
        st.caption("These are anonymized features from PCA transformation")
        
        # V features in grid layout (4 columns)
        v_features = {}
        cols = st.columns(4)
        
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with cols[col_idx]:
                v_val = st.session_state['random_data'][f'V{i}'] if st.session_state['random_data'] else 0.0
                v_features[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=float(v_val),
                    step=0.1,
                    format="%.6f",
                    key=f"v{i}"
                )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Check for Fraud", use_container_width=True, type="primary")
        
        if submitted:
            # Create dataframe with input data
            input_data = {
                'Time': time, 
                'Amount': amount,
                **v_features,
                'Time_of_day': time_of_day
            }
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time_of_day']
            input_df = input_df[expected_features]
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display results
                st.divider()
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE")
                
                with col2:
                    st.metric("Fraud Probability", f"{prediction_proba[1]*100:.2f}%")
                
                with col3:
                    st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")
                
                # Visual indicator
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK**: This transaction is predicted to be fraudulent!")
                else:
                    st.success("✅ **LOW RISK**: This transaction appears to be legitimate.")
                
                # Show probability bar
                st.write("**Fraud Probability Distribution:**")
                st.progress(float(prediction_proba[1]))
                
                # Show input data
                with st.expander("📊 View Input Data"):
                    st.dataframe(input_df.T)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure all features match the model's expected input.")

# ============================================================================
# RANDOM TRANSACTION
# ============================================================================

elif input_method == "Random Transaction":
    st.header("🎲 Random Transaction Generator")
    
    st.info("Generate a random transaction with realistic values between the 25th and 75th percentile.")
    
    if st.button("🎲 Generate and Check Random Transaction", use_container_width=True, type="primary"):
        # Generate random transaction
        random_data = generate_random_transaction()
        input_df = pd.DataFrame([random_data])
        
        # Ensure correct column order
        expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time_of_day']
        input_df = input_df[expected_features]
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.divider()
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE")
            
            with col2:
                st.metric("Fraud Probability", f"{prediction_proba[1]*100:.2f}%")
            
            with col3:
                st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")
            
            # Visual indicator
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: This transaction is predicted to be fraudulent!")
            else:
                st.success("✅ **LOW RISK**: This transaction appears to be legitimate.")
            
            # Show probability bar
            st.write("**Fraud Probability Distribution:**")
            st.progress(float(prediction_proba[1]))
            
            # Show generated data
            with st.expander("📊 View Generated Transaction Data"):
                display_df = input_df.T
                display_df.columns = ['Value']
                st.dataframe(display_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            with st.expander("Debug Info"):
                st.write("Input shape:", input_df.shape)
                st.write("Input columns:", input_df.columns.tolist())
                st.dataframe(input_df)

# ============================================================================
# CSV UPLOAD
# ============================================================================

elif input_method == "Upload CSV":
    st.header("📤 Upload Transaction CSV")
    
    st.info("Upload a CSV file with transaction data. Required columns: Time, Amount, V1-V28")
    st.caption("Note: Time_of_day will be automatically calculated if not present")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.write(f"**Loaded {len(df)} transactions**")
        st.dataframe(df.head())
        
        if st.button("🔍 Check All Transactions for Fraud", type="primary"):
            try:
                # Add Time_of_day if not present
                if 'Time_of_day' not in df.columns and 'Time' in df.columns:
                    df['Time_of_day'] = (df['Time'] / 3600) % 24
                    st.info("✅ Added Time_of_day feature")
                
                # Ensure correct column order
                expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time_of_day']
                df_features = df[expected_features]
                
                # Make predictions
                predictions = model.predict(df_features)
                prediction_probas = model.predict_proba(df_features)
                
                # Add results to dataframe
                df['Prediction'] = predictions
                df['Fraud_Probability'] = prediction_probas[:, 1]
                df['Prediction_Label'] = df['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})
                
                # Summary
                st.divider()
                st.subheader("📊 Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Transactions", len(df))
                
                with col2:
                    fraud_count = (predictions == 1).sum()
                    st.metric("Flagged as Fraud", fraud_count)
                
                with col3:
                    fraud_pct = (fraud_count / len(df)) * 100
                    st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
                
                # Show results
                st.subheader("🎯 Detailed Results")
                
                # Filter options
                filter_option = st.radio(
                    "Filter results:",
                    ["All Transactions", "Fraudulent Only", "Legitimate Only"]
                )
                
                if filter_option == "Fraudulent Only":
                    display_df = df[df['Prediction'] == 1]
                elif filter_option == "Legitimate Only":
                    display_df = df[df['Prediction'] == 0]
                else:
                    display_df = df
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                st.info("Make sure your CSV has all required columns: Time, Amount, V1-V28")

# =============================================================================
# MODEL INFORMATION PAGE
# =============================================================================

elif page == "Model Information":
    st.title("📊 Model Information & Performance")
    
    # ============================================================================
    # MODEL OVERVIEW
    # ============================================================================
    
    st.header("🔍 Model Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", type(model).__name__)
    
    with col2:
        st.metric("Total Features", "31")
    
    with col3:
        st.metric("Training Samples", "~200K")
    
    st.divider()
    
    # ============================================================================
    # FEATURE BREAKDOWN
    # ============================================================================
    
    st.header("📋 Feature Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Features")
        st.markdown("""
        - **Time**: Seconds elapsed between transactions
        - **Amount**: Transaction amount in dollars
        - **Time_of_day**: Hour of day (0-23) - engineered feature
        """)
    
    with col2:
        st.subheader("PCA Components")
        st.markdown("""
        - **V1-V28**: Principal components from PCA transformation
        - Anonymized for privacy and confidentiality
        - Capture essential transaction patterns
        """)
    
    st.divider()
    
    # ============================================================================
    # TRAINING DETAILS
    # ============================================================================
    
    st.header("⚙️ Training Configuration")
    
    # Create expandable sections
    with st.expander("🎯 Model Hyperparameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Tree Parameters:**
            - `n_estimators`: 300
            - `max_depth`: 7
            - `learning_rate`: 0.05
            """)
        
        with col2:
            st.markdown("""
            **Regularization:**
            - `min_child_weight`: 1
            - `subsample`: 0.8
            - `colsample_bytree`: 0.8
            """)
    
    with st.expander("⚖️ Class Imbalance Handling"):
        st.markdown("""
        **Challenge**: Only 0.17% of transactions are fraudulent (~492 fraud cases in 284,807 transactions)
        
        **Solutions Applied:**
        1. **SMOTE (Synthetic Minority Oversampling Technique)**
           - Generated synthetic fraud samples
           - Balanced training data to 50-50 ratio
        
        2. **Class Weights**
           - Penalized misclassifying fraud cases more heavily
           - `scale_pos_weight` parameter optimization
        
        3. **Threshold Optimization**
           - Tuned decision threshold for optimal precision-recall balance
           - Identified optimal threshold as 0.84 for fraud detection
        """)
    
    st.divider()
    
    # ============================================================================
    # PERFORMANCE METRICS
    # ============================================================================
    
    st.header("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "AUC-ROC Score",
            "0.972",
            delta="Excellent",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "F1 Score (Fraud)",
            "0.91",
            delta="Very Good",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Precision",
            "0.88",
            help="88% of flagged transactions are actually fraud"
        )
    
    with col4:
        st.metric(
            "Recall",
            "0.94",
            help="94% of fraud cases are caught"
        )
    
    st.divider()
    
    # ============================================================================
    # ROC CURVE
    # ============================================================================
    
    st.header("📉 ROC Curve Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            roc_image = Image.open("roc_curve.png")
            st.image(roc_image, use_container_width=True)
        except:
            st.warning("⚠️ ROC curve image not found. Generate with: `plot_roc_curve(model, X_test, y_test)`")
            
            # Placeholder chart
            st.info("Expected ROC curve shows strong separation between fraud and legitimate transactions")
    
    with col2:
        st.markdown("""
        **Understanding ROC-AUC:**
        
        **What it measures:**
        - Model's ability to distinguish between fraud and legitimate transactions
        
        **Score interpretation:**
        - 0.5 = Random guessing
        - 0.7-0.8 = Acceptable
        - 0.8-0.9 = Excellent
        - **0.972 = Outstanding** ⭐
        
        **Why it matters:**
        - Robust to class imbalance
        - Evaluates all possible thresholds
        - Industry standard for fraud detection
        """)
    
    st.divider()
    
    # ============================================================================
    # BUSINESS IMPACT
    # ============================================================================
    
    st.header("💼 Business Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Benefits")
        st.markdown("""
        - **94% fraud detection rate**: Catches vast majority of fraudulent transactions
        - **88% precision**: Minimizes false alarms for legitimate customers
        - **Real-time predictions**: Sub-second inference time
        - **Scalable**: Handles high transaction volumes
        - **Explainable**: SHAP values provide reasoning for each decision
        """)
    
    with col2:
        st.subheader("⚠️ Limitations")
        st.markdown("""
        - **6% missed frauds**: Some sophisticated fraud may slip through
        - **12% false positives**: Some legitimate transactions flagged
        - **Anonymized features**: Limited interpretability of V1-V28
        - **Requires retraining**: Model degrades as fraud patterns evolve
        - **Data dependency**: Performance tied to training data quality
        """)
    
    st.divider()
    
    # ============================================================================
    # FUTURE IMPROVEMENTS
    # ============================================================================
    
    st.header("🚀 Future Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Short-term (1-3 months)")
        st.markdown("""
        - [ ] Implement online learning for real-time adaptation
        - [ ] Add more temporal features (day of week, month)
        - [ ] Integrate customer behavior profiles
        - [ ] A/B test different thresholds in production
        - [ ] Build monitoring dashboard for model drift
        """)
    
    with col2:
        st.subheader("Long-term (6-12 months)")
        st.markdown("""
        - [ ] Explore deep learning approaches (LSTM, Transformers)
        - [ ] Add merchant category features
        - [ ] Implement graph-based fraud detection
        - [ ] Multi-model ensemble for higher accuracy
        - [ ] AutoML for continuous optimization
        """)
    
    st.divider()

# =============================================================================
# FEATURE IMPORTANCE PAGE
# =============================================================================

elif page == "Feature Importance":

    st.title("Feature Importance & Explainability")

    st.subheader("SHAP Summary Plot")

    try:
        shap_image = Image.open("shap_summary.png")
        st.image(shap_image, use_container_width=True)
    except:
        st.warning("SHAP image not found.")

    st.markdown("""
    SHAP values show how features influence predictions:
    - Positive values push toward fraud
    - Negative values push toward legitimate
    - Magnitude = strength of impact
    """)
