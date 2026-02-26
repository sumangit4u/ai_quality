"""
Production FastAPI Application for ADAS Model
- Loads models from Azure ML Model Registry
- Implements multi-version model serving
- Includes comprehensive logging and metrics
- Supports canary deployments and A/B testing
"""

import os
import io
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== Configuration ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['animal', 'name_board', 'other_vehicle', 'pedestrian', 'pothole', 'road_sign', 'speed_breaker']
NUM_CLASSES = len(CLASS_NAMES)
CANARY_PERCENTAGE = 30  # Traffic split between v1 and v2

# Azure ML Configuration (will be set from environment)
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "")
AZURE_ML_SUBSCRIPTION = os.getenv("AZURE_ML_SUBSCRIPTION", "")
AZURE_ML_RESOURCE_GROUP = os.getenv("AZURE_ML_RESOURCE_GROUP", "")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./models")

# ======================== Models ========================

class BaselineModel(nn.Module):
    """Version 1: Basic ResNet-18 without regularization"""
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class DropoutModel(nn.Module):
    """Version 2: ResNet-18 with Dropout for regularization"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


# ======================== Model Loading ========================

def load_models_from_azure():
    """
    Load models from Azure ML Registry
    In production, this would use MLflow or azureml-core
    For demo, loads from local path
    """
    try:
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        
        model_v1 = BaselineModel(NUM_CLASSES).to(DEVICE)
        model_v2 = DropoutModel(NUM_CLASSES).to(DEVICE)
        
        # In production, download from Azure ML:
        # from azureml.core.model import Model
        # model_path_v1 = Model.get_model_path('adas-model-v1')
        # model_v1.load_state_dict(torch.load(model_path_v1))
        
        model_v1.eval()
        model_v2.eval()
        
        logger.info(f"Models loaded on device: {DEVICE}")
        logger.info("Note: Using random weights for demo. In production, load from Azure ML Registry.")
        
        return model_v1, model_v2
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


# ======================== API Schemas ========================

class PredictionResponse(BaseModel):
    """Response for single model prediction"""
    image_id: str
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str
    latency_ms: float


class ComparisonResponse(BaseModel):
    """Response for comparing two models (A/B testing)"""
    image_id: str
    v1_prediction: str
    v1_confidence: float
    v2_prediction: str
    v2_confidence: float
    agreement: bool
    v1_latency_ms: float
    v2_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_version: str
    device: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    num_classes: int
    classes: List[str]
    input_shape: str
    device: str


class MetricsResponse(BaseModel):
    """Aggregated metrics response"""
    total_requests: int
    v1_requests: int
    v2_requests: int
    avg_latency_ms: float
    agreement_rate: float
    error_rate: float


# ======================== Logging System ========================

class PredictionLog:
    """Store prediction metadata for analysis"""
    def __init__(self, timestamp: str, image_id: str, model_version: str,
                 prediction: str, confidence: float, latency_ms: float,
                 status: str = "success", error_msg: Optional[str] = None):
        self.timestamp = timestamp
        self.image_id = image_id
        self.model_version = model_version
        self.prediction = prediction
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.status = status
        self.error_msg = error_msg
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'image_id': self.image_id,
            'model_version': self.model_version,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'status': self.status,
            'error_msg': self.error_msg
        }


# ======================== FastAPI App Setup ========================

# Load models
model_v1, model_v2 = load_models_from_azure()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction logs for tracking
prediction_logs: List[PredictionLog] = []

# Create FastAPI app
app = FastAPI(
    title="ADAS Model API",
    description="Production CNN model for detecting ADAS objects with multi-version support",
    version="2.0.0"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application initialized")


# ======================== Utility Functions ========================

def validate_image_file(file_ext: str) -> bool:
    """Validate file extension"""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return file_ext.lower() in allowed_extensions


def process_image(contents: bytes) -> Image.Image:
    """Load and validate image"""
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Could not open image: {str(e)}")


def validate_image_dimensions(image: Image.Image, min_size: int = 32) -> bool:
    """Validate image dimensions"""
    return image.size[0] >= min_size and image.size[1] >= min_size


# ======================== API Endpoints ========================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_version="v2.0",
        device=str(DEVICE),
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    total_params = sum(p.numel() for p in model_v1.parameters())
    return ModelInfoResponse(
        model_name="ResNet-18 ADAS Detector",
        version="2.0",
        num_classes=NUM_CLASSES,
        classes=CLASS_NAMES,
        input_shape="128x128x3",
        device=str(DEVICE)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_with_routing(
    file: UploadFile = File(...),
    model_version: Optional[str] = Query(None, description="Force specific model (v1 or v2). If None, uses canary split.")
):
    """
    Predict class of uploaded image with optional version selection.
    
    Args:
        file: Image file (JPEG, PNG, GIF, BMP)
        model_version: Optional force to v1 or v2. Default uses canary split.
    
    Returns:
        PredictionResponse with prediction and metrics
    """
    start_time = time.time()
    image_id = f"{datetime.now().timestamp()}"
    
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        
        if not validate_image_file(file_ext):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_ext}. Allowed: jpg, jpeg, png, gif, bmp"
            )
        
        # Read image - MUST await and only call once
        try:
            contents = await file.read()
        except Exception as read_error:
            logger.error(f"Error reading file: {str(read_error)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_error)}")
        
        # Check if file is empty
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file - no data received")
        
        # Process image
        try:
            image = process_image(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate dimensions
        if not validate_image_dimensions(image):
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {image.size}. Minimum: 32x32"
            )
        
        # Route decision
        if model_version and model_version in ['v1', 'v2']:
            use_v2 = model_version == 'v2'
        else:
            use_v2 = random.random() < (CANARY_PERCENTAGE / 100)
        
        model = model_v2 if use_v2 else model_v1
        internal_version = "v2.0" if use_v2 else "v1.0"
        
        # Transform and predict
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Build class probabilities dict
        class_probs = {
            class_name: float(probabilities[0, idx].item())
            for idx, class_name in enumerate(CLASS_NAMES)
        }
        
        latency = (time.time() - start_time) * 1000
        
        # Log prediction
        log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            image_id=image_id,
            model_version=internal_version,
            prediction=predicted_class,
            confidence=round(confidence_value, 4),
            latency_ms=round(latency, 2),
            status="success"
        )
        prediction_logs.append(log)
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence_value:.4f}, version: {internal_version})")
        
        return PredictionResponse(
            image_id=image_id,
            prediction=predicted_class,
            confidence=round(confidence_value, 4),
            class_probabilities=class_probs,
            model_version=internal_version,
            latency_ms=round(latency, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-both", response_model=ComparisonResponse)
async def predict_both_models(file: UploadFile = File(...)):
    """
    Get predictions from BOTH models for A/B testing comparison.
    
    Returns:
        ComparisonResponse showing both predictions and agreement
    """
    image_id = f"{datetime.now().timestamp()}"
    
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        
        if not validate_image_file(file_ext):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_ext}"
            )
        
        # Read image - MUST await and only call once
        try:
            contents = await file.read()
            logger.info(f"File received for A/B test: {file.filename}, Size: {len(contents) if contents else 0} bytes")
        except Exception as read_error:
            logger.error(f"Error reading file in A/B test: {str(read_error)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_error)}")
        
        # Check if file is empty
        if not contents or len(contents) == 0:
            logger.error(f"File is empty after read in A/B test")
            raise HTTPException(status_code=400, detail="Empty file - no data received")
        
        # Process image
        try:
            image = process_image(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate dimensions
        if not validate_image_dimensions(image):
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {image.size}"
            )
        
        # Transform
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # V1 Prediction
        start_v1 = time.time()
        with torch.no_grad():
            outputs_v1 = model_v1(image_tensor)
            probs_v1 = torch.softmax(outputs_v1, dim=1)
            conf_v1, idx_v1 = torch.max(probs_v1, 1)
        latency_v1 = (time.time() - start_v1) * 1000
        
        # V2 Prediction
        start_v2 = time.time()
        with torch.no_grad():
            outputs_v2 = model_v2(image_tensor)
            probs_v2 = torch.softmax(outputs_v2, dim=1)
            conf_v2, idx_v2 = torch.max(probs_v2, 1)
        latency_v2 = (time.time() - start_v2) * 1000
        
        # Get predictions
        pred_v1 = CLASS_NAMES[idx_v1.item()]
        pred_v2 = CLASS_NAMES[idx_v2.item()]
        agreement = pred_v1 == pred_v2
        
        # Log both
        for model_name, pred, conf, latency in [
            ("v1.0", pred_v1, conf_v1.item(), latency_v1),
            ("v2.0", pred_v2, conf_v2.item(), latency_v2)
        ]:
            log = PredictionLog(
                timestamp=datetime.now().isoformat(),
                image_id=image_id,
                model_version=model_name,
                prediction=pred,
                confidence=round(conf, 4),
                latency_ms=round(latency, 2),
                status="success"
            )
            prediction_logs.append(log)
        
        logger.info(f"A/B Comparison: V1={pred_v1}, V2={pred_v2}, Agreement={agreement}")
        
        return ComparisonResponse(
            image_id=image_id,
            v1_prediction=pred_v1,
            v1_confidence=round(conf_v1.item(), 4),
            v2_prediction=pred_v2,
            v2_confidence=round(conf_v2.item(), 4),
            agreement=agreement,
            v1_latency_ms=round(latency_v1, 2),
            v2_latency_ms=round(latency_v2, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in A/B comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get aggregated performance metrics from prediction logs
    """
    if not prediction_logs:
        return MetricsResponse(
            total_requests=0,
            v1_requests=0,
            v2_requests=0,
            avg_latency_ms=0.0,
            agreement_rate=0.0,
            error_rate=0.0
        )
    
    df = pd.DataFrame([log.to_dict() for log in prediction_logs])
    
    total_requests = len(df)
    success_requests = len(df[df['status'] == 'success'])
    v1_requests = len(df[df['model_version'] == 'v1.0'])
    v2_requests = len(df[df['model_version'] == 'v2.0'])
    avg_latency = df[df['status'] == 'success']['latency_ms'].mean() if success_requests > 0 else 0
    error_rate = ((total_requests - success_requests) / total_requests * 100) if total_requests > 0 else 0
    
    # Agreement rate
    agreement_count = 0
    comparison_pairs = 0
    
    for image_id in df['image_id'].unique():
        logs_for_image = df[df['image_id'] == image_id]
        if len(logs_for_image) == 2:
            v1_logs = logs_for_image[logs_for_image['model_version'] == 'v1.0']
            v2_logs = logs_for_image[logs_for_image['model_version'] == 'v2.0']
            if not v1_logs.empty and not v2_logs.empty:
                v1_pred = v1_logs.iloc[0]['prediction']
                v2_pred = v2_logs.iloc[0]['prediction']
                if v1_pred == v2_pred:
                    agreement_count += 1
                comparison_pairs += 1
    
    agreement_rate = (agreement_count / comparison_pairs * 100) if comparison_pairs > 0 else 0
    
    return MetricsResponse(
        total_requests=total_requests,
        v1_requests=v1_requests,
        v2_requests=v2_requests,
        avg_latency_ms=round(avg_latency, 2),
        agreement_rate=round(agreement_rate, 2),
        error_rate=round(error_rate, 2)
    )


@app.get("/logs")
async def get_logs(limit: int = Query(100, ge=1, le=1000)):
    """
    Get prediction logs (limited for performance)
    
    Args:
        limit: Maximum number of logs to return
    """
    logs_dict = [log.to_dict() for log in prediction_logs[-limit:]]
    return {
        "total_logs": len(prediction_logs),
        "returned_logs": len(logs_dict),
        "logs": logs_dict
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    if not prediction_logs:
        return {"message": "No predictions logged yet"}
    
    df = pd.DataFrame([log.to_dict() for log in prediction_logs])
    
    stats = {
        "total_requests": len(df),
        "total_unique_images": df['image_id'].nunique(),
        "success_rate": round((len(df[df['status'] == 'success']) / len(df) * 100), 2),
        "by_model_version": df['model_version'].value_counts().to_dict(),
        "by_prediction": df['prediction'].value_counts().to_dict(),
        "latency_stats": {
            "mean_ms": round(df['latency_ms'].mean(), 2),
            "min_ms": round(df['latency_ms'].min(), 2),
            "max_ms": round(df['latency_ms'].max(), 2),
            "std_ms": round(df['latency_ms'].std(), 2)
        },
        "confidence_stats": {
            "mean": round(df['confidence'].mean(), 4),
            "min": round(df['confidence'].min(), 4),
            "max": round(df['confidence'].max(), 4)
        }
    }
    
    return stats


# ======================== Main ========================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
