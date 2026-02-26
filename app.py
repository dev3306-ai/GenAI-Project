import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

from models.forecaster import SolarForecaster
from utils.preprocessing import load_and_preprocess, engineer_features, scale_features

st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

st.title("☀️ Intelligent Solar Energy Forecasting System")
st.markdown("### ML-Based Solar Power Generation Analytics")

# Sidebar
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload Solar Data CSV", type=['csv'])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully!")

if df is not None:
    # Display data info
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Date Range", f"{len(df)//24} days")
    
    with st.expander("View Raw Data"):
        st.dataframe(df.head(50))
    
    # Preprocessing
    st.subheader("🔧 Data Preprocessing")
    df_processed = load_and_preprocess(df)
    
    # Feature selection first to know target
    st.subheader("🎯 Model Configuration")
    
    target_col = st.selectbox("Target Variable", 
                              [col for col in df_processed.columns if 'power' in col.lower()],
                              index=0 if any('power' in col.lower() for col in df_processed.columns) else None)
    
    # Now engineer features with the target column
    if target_col:
        df_processed = engineer_features(df_processed, target_col)
    else:
        df_processed = engineer_features(df_processed)
    
    # Show preprocessing stats
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values Handled:** ✓")
        st.write("**Temporal Features Extracted:** ✓")
    with col2:
        st.write("**Feature Engineering Applied:** ✓")
        st.write(f"**Total Features:** {len(df_processed.columns)}")
    
    # Continue with feature selection
    
    # Auto-select numeric features
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    feature_cols = st.multiselect("Select Features", numeric_cols, default=numeric_cols[:5])
    
    model_type = st.selectbox("Model Type", 
                             ["random_forest", "gradient_boosting", "linear_regression"])
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    if st.button("🚀 Train Model", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature!")
        else:
            with st.spinner("Training model..."):
                # Prepare data
                X = df_processed[feature_cols].values
                y = df_processed[target_col].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
                
                # Train model
                forecaster = SolarForecaster(model_type=model_type)
                forecaster.train(X_train_scaled, y_train)
                
                # Evaluate
                results = forecaster.evaluate(X_test_scaled, y_test)
                
                # Store in session state
                st.session_state['model'] = forecaster
                st.session_state['results'] = results
                st.session_state['scaler'] = scaler
                st.session_state['feature_cols'] = feature_cols
                st.session_state['y_test'] = y_test
                
                st.success("✅ Model trained successfully!")
    
    # Display results
    if 'results' in st.session_state:
        st.subheader("📈 Model Performance")
        
        results = st.session_state['results']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{results['MAE']:.2f}")
        col2.metric("RMSE", f"{results['RMSE']:.2f}")
        col3.metric("R² Score", f"{results['R2']:.3f}")
        
        # Prediction vs Actual plot
        st.subheader("🔍 Predictions vs Actual")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state['y_test'][:200],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            y=results['predictions'][:200],
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="Energy Generation: Actual vs Predicted",
            xaxis_title="Sample Index",
            yaxis_title="Power Generated",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if model_type in ['random_forest', 'gradient_boosting']:
            st.subheader("🎯 Feature Importance")
            importance = st.session_state['model'].get_feature_importance(feature_cols)
            if importance:
                imp_df = pd.DataFrame(list(importance.items()), 
                                     columns=['Feature', 'Importance'])
                imp_df = imp_df.sort_values('Importance', ascending=False)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecast insights
        st.subheader("💡 Key Insights")
        st.write(f"- Model Type: **{model_type.replace('_', ' ').title()}**")
        st.write(f"- Training Samples: **{len(X_train_scaled)}**")
        st.write(f"- Test Samples: **{len(X_test_scaled)}**")
        st.write(f"- Average Prediction Error: **±{results['MAE']:.2f} units**")
        
        if results['R2'] > 0.8:
            st.success("🎉 Excellent model performance!")
        elif results['R2'] > 0.6:
            st.info("✓ Good model performance")
        else:
            st.warning("⚠️ Model may need improvement")

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    
    st.markdown("""
    ### About This System
    
    This intelligent solar energy forecasting system uses machine learning to predict solar power generation.
    
    **Required CSV Columns:**
    - `datetime` or `date` - Timestamp
    - `irradiance` - Solar radiation (W/m²)
    - `temperature` - Ambient temperature (°C)
    - `cloud_cover` - Cloud coverage (%)
    - `power_generated` - Solar power output (target variable)
    
    **Supported Models:**
    - Random Forest Regression
    - Gradient Boosting Regression
    - Linear Regression
    
    **Features:**
    - Automated data preprocessing
    - Feature engineering (temporal features, rolling averages)
    - Multiple ML algorithms
    - Performance evaluation (MAE, RMSE, R²)
    - Interactive visualizations
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**Solar Energy Forecasting v1.0**")
st.sidebar.markdown("GenAI Capstone - Milestone 1")
