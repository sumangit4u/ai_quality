import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import io
from PIL import Image

# ======================== Configuration ========================

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ADAS Model Inference",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 ADAS Model Inference Dashboard")
st.markdown("---")

# ======================== Sidebar Navigation ========================

page = st.sidebar.radio(
    "Navigation",
    ["📸 Single Prediction", "🔄 A/B Testing", "📊 Metrics Dashboard", "📝 Logs"]
)

# ======================== Helper Functions ========================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_single(uploaded_file):
    """Send single image to /predict endpoint"""
    if uploaded_file is None:
        st.error("❌ No file uploaded")
        return None
    
    try:
        # CRITICAL: Read file bytes ONCE and wrap in BytesIO
        file_bytes = uploaded_file.read()
        
        if not file_bytes or len(file_bytes) == 0:
            st.error("❌ File is empty - please upload a valid image")
            return None
        
        st.info(f"📤 Uploading {uploaded_file.name} ({len(file_bytes)} bytes)...")
        
        # Send to API with BytesIO wrapper
        files = {
            'file': (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
            return None
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def predict_both(uploaded_file):
    """Send image to /predict-both endpoint for A/B testing"""
    if uploaded_file is None:
        st.error("❌ No file uploaded")
        return None
    
    try:
        # CRITICAL: Read file bytes ONCE and wrap in BytesIO
        file_bytes = uploaded_file.read()
        
        if not file_bytes or len(file_bytes) == 0:
            st.error("❌ File is empty - please upload a valid image")
            return None
        
        st.info(f"📤 Uploading {uploaded_file.name} for A/B testing ({len(file_bytes)} bytes)...")
        
        # Send to API with BytesIO wrapper
        files = {
            'file': (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict-both",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
            return None
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# ======================== Page: Single Prediction ========================

if page == "📸 Single Prediction":
    st.header("Single Image Prediction")
    
    # Check API health
    if not check_api_health():
        st.error("🔴 API is not running! Start it with: `python api.py`")
    else:
        st.success("🟢 API is online")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            key="single_upload"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        if uploaded_file is not None:
            # Reset file pointer for display
            uploaded_file.seek(0)
            
            if st.button("🔮 Predict", key="single_predict"):
                result = predict_single(uploaded_file)
                
                if result:
                    st.success("✅ Prediction successful!")
                    
                    # Display main prediction
                    col_pred, col_conf = st.columns(2)
                    with col_pred:
                        st.metric("Predicted Class", result['prediction'])
                    with col_conf:
                        st.metric("Confidence", f"{result['confidence']:.4f}")
                    
                    # Display model info
                    col_model, col_latency = st.columns(2)
                    with col_model:
                        st.metric("Model Version", result['model_version'])
                    with col_latency:
                        st.metric("Latency", f"{result['latency_ms']:.2f} ms")
                    
                    # Display all probabilities
                    st.subheader("Class Probabilities")
                    probs_df = pd.DataFrame(
                        list(result['class_probabilities'].items()),
                        columns=['Class', 'Probability']
                    ).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(probs_df.set_index('Class'))
                    
                    # Show as table
                    st.dataframe(probs_df, use_container_width=True)
                    
                    # JSON export
                    st.json(result)


# ======================== Page: A/B Testing ========================

elif page == "🔄 A/B Testing":
    st.header("Model A/B Testing (v1.0 vs v2.0)")
    st.markdown("Compare predictions from both models side-by-side")
    
    if not check_api_health():
        st.error("🔴 API is not running!")
    else:
        st.success("🟢 API is online")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for A/B testing",
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            key="ab_upload"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Comparison Results")
        
        if uploaded_file is not None:
            # Reset file pointer for display
            uploaded_file.seek(0)
            
            if st.button("🔄 Compare Models", key="ab_predict"):
                result = predict_both(uploaded_file)
                
                if result:
                    st.success("✅ Comparison complete!")
                    
                    # Agreement badge
                    if result['agreement']:
                        st.success("✅ Models agree!")
                    else:
                        st.warning("⚠️ Models disagree")
                    
                    # Model comparison table
                    comparison_data = {
                        'Metric': ['Prediction', 'Confidence', 'Latency (ms)'],
                        'v1.0': [
                            result['v1_prediction'],
                            f"{result['v1_confidence']:.4f}",
                            f"{result['v1_latency_ms']:.2f}"
                        ],
                        'v2.0': [
                            result['v2_prediction'],
                            f"{result['v2_confidence']:.4f}",
                            f"{result['v2_latency_ms']:.2f}"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                    
                    # Confidence comparison
                    st.subheader("Confidence Comparison")
                    conf_data = {
                        'Model': ['v1.0', 'v2.0'],
                        'Confidence': [result['v1_confidence'], result['v2_confidence']]
                    }
                    st.bar_chart(pd.DataFrame(conf_data).set_index('Model'))
                    
                    # JSON export
                    st.json(result)


# ======================== Page: Metrics Dashboard ========================

elif page == "📊 Metrics Dashboard":
    st.header("API Metrics Dashboard")
    
    if not check_api_health():
        st.error("🔴 API is not running!")
    else:
        st.success("🟢 API is online")
    
    if st.button("🔄 Refresh Metrics"):
        pass
    
    try:
        # Get metrics
        metrics_response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", metrics['total_requests'])
            with col2:
                st.metric("V1 Requests", metrics['v1_requests'])
            with col3:
                st.metric("V2 Requests", metrics['v2_requests'])
            with col4:
                st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.2f} ms")
            
            # Additional metrics
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric("Agreement Rate", f"{metrics['agreement_rate']:.2f}%")
            with col6:
                st.metric("Error Rate", f"{metrics['error_rate']:.2f}%")
            
            # Detailed stats
            st.subheader("Detailed Statistics")
            stats_response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                if "total_requests" in stats:
                    st.json(stats)
                else:
                    st.info("No statistics available yet")
        else:
            st.error("Failed to fetch metrics")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ======================== Page: Logs ========================

elif page == "📝 Logs":
    st.header("Prediction Logs")
    
    if not check_api_health():
        st.error("🔴 API is not running!")
    else:
        st.success("🟢 API is online")
    
    limit = st.slider("Number of logs to display", 10, 500, 100)
    
    if st.button("🔄 Refresh Logs"):
        pass
    
    try:
        logs_response = requests.get(f"{API_BASE_URL}/logs?limit={limit}", timeout=5)
        
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            
            st.info(f"Showing {logs_data['returned_logs']} of {logs_data['total_logs']} logs")
            
            if logs_data['returned_logs'] > 0:
                logs_df = pd.DataFrame(logs_data['logs'])
                
                # Display table
                st.dataframe(logs_df, use_container_width=True, height=400)
                
                # Download as CSV
                csv = logs_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"adas_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No logs available yet. Make some predictions!")
        else:
            st.error("Failed to fetch logs")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")