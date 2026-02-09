# src/stream.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import logging
from pipelines.base_pipeline import base_pipeline, train_more_pipeline, predict_pipeline
import pickle
from io import BytesIO
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Incremental Learning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .active-model-box {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_model_path' not in st.session_state:
    st.session_state.active_model_path = None
if 'active_model_info' not in st.session_state:
    st.session_state.active_model_info = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def display_active_model():
    """Display information about the currently active model."""
    if st.session_state.active_model_path and st.session_state.active_model_info:
        info = st.session_state.active_model_info
        st.markdown(f"""
        <div class="active-model-box">
            <h4>🎯 Active Model</h4>
            <p><strong>Type:</strong> {info['model_type']}</p>
            <p><strong>Target:</strong> {info['target']}</p>
            <p><strong>Score:</strong> {info['score']:.4f}</p>
            <p><strong>Status:</strong> ✅ Ready for retraining or predictions</p>
            <p><strong>Removed Columns:</strong> {info['ids']}</p>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False

def train_new_model_tab():
    """Tab for training a new model."""
    st.markdown('<div class="section-header">📚 Train New Model</div>', unsafe_allow_html=True)
    
    # Show if there's an active model
    if display_active_model():
        st.info("💡 You have an active model. You can train a new one to replace it, or use the other tabs to retrain/predict.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Training Data (CSV or XLSX)",
            type=['csv', 'xlsx'],
            key="train_file"
        )
        
        if uploaded_file is not None:
            # Display data preview
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head(10))
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Rows", df.shape[0])
                    with col_b:
                        st.metric("Total Columns", df.shape[1])
                    with col_c:
                        st.metric("Missing Values", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    with col2:
        st.markdown("### Model Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ['RandomForestRegressor', 'RandomForestClassifier'],
            help="Choose the algorithm for your task"
        )
        
        # Target column
        if uploaded_file is not None:
            target_column = st.selectbox(
                "Target Column",
                options=df.columns.tolist(),
                help="Select the column you want to predict"
            )
            
            # Identifier columns (optional)
            id_columns = st.multiselect(
                "Identifier Columns (Optional)",
                options=df.columns.tolist(),
                help="Columns to exclude from training (e.g., ID, Name)"
            )
            
            # Model export name
            model_name = st.text_input(
                "Model Name",
                value="trained_model",
                help="Name for the saved model file"
            )
    
    # Train button
    if uploaded_file is not None:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a while."):
                try:
                    # Save uploaded file
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    # Create export path
                    export_path = f"models/{model_name}.pkl"
                    os.makedirs("models", exist_ok=True)
                    
                    buffer_path = f"models/{model_name}_buffer.pkl"
                    # Train model
                    score = base_pipeline(
                        source=temp_file_path,
                        model=model_type,
                        target=target_column,
                        export_path=export_path,
                        buffer_path=buffer_path,
                        ids=id_columns if id_columns else None
                    )
                    
                    # Update session state - set as active model
                    st.session_state.active_model_path = export_path
                    st.session_state.buffer_path = buffer_path
                    st.session_state.active_model_info = {
                        'model_type': model_type,
                        'score': score,
                        'target': target_column,
                        'path': export_path,
                        'name': model_name, 
                        'ids': id_columns if id_columns else []
                    }
                    
                    # Add to history
                    st.session_state.training_history.append({
                        'model_type': model_type,
                        'score': score,
                        'target': target_column,
                        'path': export_path,
                        'name': model_name, 
                        'ids': id_columns if id_columns else [],
                        'buffer_path': buffer_path
                    })
                    
                    # Clean up
                    os.unlink(temp_file_path)
                    
                    # Display success
                    st.success(f"✅ Model trained successfully and set as active!")
                    st.balloons()
                    st.metric("Model Score", f"{score:.4f}")
                    
                    st.info("🎉 Your model is now active! Go to 'Retrain Model' or 'Make Predictions' tabs to use it.")
                    
                    # Download model
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Trained Model",
                            data=f.read(),
                            file_name=f"{model_name}.pkl",
                            mime="application/octet-stream"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    logger.error(f"Training error: {e}")

def retrain_model_tab():
    """Tab for retraining an existing model."""
    st.markdown('<div class="section-header">🔄 Retrain Existing Model</div>', unsafe_allow_html=True)
    
    # Check if there's an active model
    has_active_model = display_active_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload new training data
        new_data_file = st.file_uploader(
            "Upload New Training Data (CSV or XLSX)",
            type=['csv', 'xlsx'],
            key="retrain_data_file"
        )
        
        if new_data_file is not None:
            try:
                if new_data_file.name.endswith('.csv'):
                    df = pd.read_csv(new_data_file)
                else:
                    df = pd.read_excel(new_data_file)
                
                st.success(f"✅ New data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    with col2:
        st.markdown("### Retraining Configuration")
        
        # Model source selection
        model_source = st.radio(
            "Select Model Source",
            options=["Use Active Model", "Upload New Model"],
            help="Choose whether to use the active model or upload a different one"
        )
        
        model_file = None
        model_path_to_use = None
        
        if model_source == "Use Active Model":
            if has_active_model:
                st.success("✅ Using active model")
                model_path_to_use = st.session_state.active_model_path
            else:
                st.warning("⚠️ No active model available. Please train a model first or upload one.")
        else:
            # Upload existing model
            model_file = st.file_uploader(
                "Upload Model (.pkl)",
                type=['pkl'],
                key="retrain_model_file"
            )
            if model_file:
                st.success("✅ Model uploaded")
        
        if new_data_file is not None:
            # Target column
            target_column = st.selectbox(
                "Target Column",
                options=df.columns.tolist(),
                key="retrain_target"
            )
            
            # Identifier columns
            id_columns = st.multiselect(
                "Identifier Columns (Optional)",
                options=df.columns.tolist(),
                key="retrain_ids"
            )
    
    # Retrain button
    can_retrain = new_data_file is not None and (model_path_to_use is not None or model_file is not None)
    
    if can_retrain:
        if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
            with st.spinner("Retraining model..."):
                try:
                    # Save data file
                    temp_data_path = save_uploaded_file(new_data_file)
                    
                    # Determine model path
                    if model_source == "Use Active Model":
                        temp_model_path = model_path_to_use
                        temp_buffer_path = st.session_state.active_buffer_path if 'active_buffer_path' in st.session_state else None
                        cleanup_model = False
                    else:
                        temp_model_path = save_uploaded_file(model_file)
                        temp_buffer_path = None
                        cleanup_model = True
                    
                    # Retrain
                    score = train_more_pipeline(
                        source=temp_data_path,
                        trained_model_path=temp_model_path,
                        buffer_path=temp_buffer_path if temp_buffer_path else st.session_state.buffer_path,
                        target=target_column,
                        ids=id_columns if id_columns else None
                    )
                    
                    # Update active model info
                    if st.session_state.active_model_info:
                        st.session_state.active_model_info['score'] = score
                        st.session_state.active_model_info['target'] = target_column
                    
                    # Clean up data file
                    os.unlink(temp_data_path)
                    if cleanup_model and temp_model_path:
                        os.unlink(temp_model_path)
                    
                    # Display success
                    st.success(f"✅ Model retrained successfully!")
                    st.balloons()
                    st.metric("Updated Model Score", f"{score:.4f}")
                    
                    # Download updated model
                    with open(st.session_state.active_model_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Updated Model",
                            data=f.read(),
                            file_name="retrained_model.pkl",
                            mime="application/octet-stream",
                            key="download_retrained"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Retraining failed: {e}")
                    logger.error(f"Retraining error: {e}")
    elif new_data_file is None:
        st.info("📤 Please upload training data to retrain the model")
    elif not has_active_model and model_source == "Use Active Model":
        st.warning("⚠️ No active model. Please train a model first or upload one.")

def predict_tab():
    """Tab for making predictions."""
    st.markdown('<div class="section-header">🎯 Make Predictions</div>', unsafe_allow_html=True)
    
    # Check if there's an active model
    has_active_model = display_active_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload prediction data
        predict_file = st.file_uploader(
            "Upload Data for Predictions (CSV or XLSX)",
            type=['csv', 'xlsx'],
            key="predict_file"
        )
        
        if predict_file is not None:
            try:
                if predict_file.name.endswith('.csv'):
                    df = pd.read_csv(predict_file)
                else:
                    df = pd.read_excel(predict_file)
                
                st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
    
    with col2:
        st.markdown("### Prediction Configuration")
        
        # Model source selection
        model_source = st.radio(
            "Select Model Source",
            options=["Use Active Model", "Upload Model"],
            help="Choose whether to use the active model or upload a different one",
            key="predict_model_source"
        )
        
        model_file = None
        model_path_to_use = None
        
        if model_source == "Use Active Model":
            if has_active_model:
                st.success("✅ Using active model")
                model_path_to_use = st.session_state.active_model_path
            else:
                st.warning("⚠️ No active model available. Please train a model first or upload one.")
        else:
            # Upload trained model
            model_file = st.file_uploader(
                "Upload Trained Model (.pkl)",
                type=['pkl'],
                key="predict_model_file"
            )
            if model_file:
                st.success("✅ Model uploaded")
    
    # Predict button
    can_predict = predict_file is not None and (model_path_to_use is not None or model_file is not None)
    
    if can_predict:
        if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                try:
                    # Save data file
                    temp_data_path = save_uploaded_file(predict_file)
                    
                    # Determine model path
                    if model_source == "Use Active Model":
                        temp_model_path = model_path_to_use
                        cleanup_model = False
                    else:
                        temp_model_path = save_uploaded_file(model_file)
                        cleanup_model = True
                    
                    ids_to_remove = []

                    if model_source == "Use Active Model" and st.session_state.active_model_info:
                        ids_to_remove = st.session_state.active_model_info.get('ids', [])
                    # Make predictions
                    predictions_df = predict_pipeline(
                        data_source=temp_data_path,
                        trained_model_path=temp_model_path,
                        ids=ids_to_remove if ids_to_remove else None
                    )
                    
                    # Store in session state
                    st.session_state.predictions = predictions_df
                    
                    # Clean up
                    os.unlink(temp_data_path)
                    if cleanup_model and temp_model_path:
                        os.unlink(temp_model_path)
                    
                    # Display results
                    st.success(f"✅ Predictions generated successfully!")
                    
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Statistics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Predictions", len(predictions_df))
                    with col_b:
                        if predictions_df['prediction'].dtype in ['int64', 'float64']:
                            st.metric("Mean Prediction", f"{predictions_df['prediction'].mean():.2f}")
                    
                    # Download predictions
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}")
    elif predict_file is None:
        st.info("📤 Please upload data to generate predictions")
    elif not has_active_model and model_source == "Use Active Model":
        st.warning("⚠️ No active model. Please train a model first or upload one.")

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🤖 Incremental Learning Platform</div>', unsafe_allow_html=True)
    st.markdown("Train, retrain, and deploy machine learning models with incremental learning capabilities.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        **Workflow:**
        1. **Train** a new model (Tab 1)
        2. Model becomes **active** automatically
        3. **Retrain** with new data (Tab 2)
        4. **Predict** on new data (Tab 3)
        
        No need to re-upload models between steps!
        """)
        
        st.markdown("---")
        
        # Active model status
        if st.session_state.active_model_info:
            st.markdown("### 🎯 Active Model Status")
            info = st.session_state.active_model_info
            st.markdown(f"""
            <div class="info-box">
                <p><strong>Name:</strong> {info.get('name', 'N/A')}</p>
                <p><strong>Type:</strong> {info['model_type']}</p>
                <p><strong>Score:</strong> {info['score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to clear active model
            if st.button("🗑️ Clear Active Model", use_container_width=True):
                st.session_state.active_model_path = None
                st.session_state.active_model_info = None
                st.rerun()
        else:
            st.info("No active model. Train one to get started!")
        
        st.markdown("---")
        
        # Training history
        if st.session_state.training_history:
            st.markdown("### 📈 Training History")
            for i, record in enumerate(reversed(st.session_state.training_history[-5:])):
                with st.expander(f"Model: {record.get('name', f'Model {i+1}')}", expanded=False):
                    st.text(f"Type: {record['model_type']}")
                    st.text(f"Score: {record['score']:.4f}")
                    st.text(f"Target: {record['target']}")
                    
                    # Option to set as active
                    if st.button(f"Set as Active", key=f"activate_{i}"):
                        st.session_state.active_model_path = record['path']
                        st.session_state.active_model_info = record
                        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎓 Train New Model", "🔄 Retrain Model", "🎯 Make Predictions"])
    
    with tab1:
        train_new_model_tab()
    
    with tab2:
        retrain_model_tab()
    
    with tab3:
        predict_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Powered by River ML & Streamlit | Incremental Learning Platform v1.0</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()