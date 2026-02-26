# ADAS Model API - Production Deployment Guide

Complete production-ready solution for serving ADAS (Advanced Driver Assistance System) CNN models with multi-version support, canary deployments, and A/B testing.

## 🎯 Overview

This project transforms the notebook examples from Class 1 and Class 2 into a **production-grade deployment** with:

- **FastAPI Backend**: Multi-model serving with version management
- **Streamlit Frontend**: User-friendly web interface
- **Docker Containerization**: Easy deployment and scaling
- **Azure ML Integration**: Model registry and cloud deployment
- **Comprehensive Testing**: API and integration test suites
- **Monitoring & Logging**: Built-in metrics and prediction tracking

## 📁 Project Structure

```
class2/
├── api.py                      # FastAPI application
├── streamlit_app.py            # Streamlit frontend
├── test_api.py                 # API test suite
├── test_integration.py         # Integration tests
├── requirements-api.txt        # API dependencies
├── requirements-streamlit.txt  # Streamlit dependencies
├── Dockerfile                  # API container image
├── Dockerfile.streamlit        # Streamlit container image
├── docker-compose.yml          # Multi-container orchestration
├── azure_setup.ps1             # Azure setup (Windows PowerShell)
├── azure_setup.sh              # Azure setup (Bash)
├── .env.example                # Environment configuration template
├── README.md                   # This file
└── models/                     # Local model storage
    ├── v1.0/                   # V1 model weights
    └── v2.0/                   # V2 model weights
```

## 🚀 Quick Start

### Local Development (No Docker)

#### 1. Install Dependencies

```bash
cd class2
pip install -r requirements-api.txt
pip install -r requirements-streamlit.txt
```

#### 2. Start API Server

```bash
# Terminal 1
python api.py
```

API will be available at: `http://localhost:8000`
API docs (Swagger): `http://localhost:8000/docs`

#### 3. Start Streamlit App

```bash
# Terminal 2
streamlit run streamlit_app.py
```

Streamlit will be available at: `http://localhost:8501`

### Docker Compose (Recommended for Development)

#### 1. Build and Run

```bash
docker-compose up -d --build
```

#### 2. Access Services

- **API**: `http://localhost:8000/docs`
- **Streamlit**: `http://localhost:8501`

#### 3. View Logs

```bash
docker-compose logs -f api
docker-compose logs -f streamlit
```

#### 4. Stop Services

```bash
docker-compose down
```

## 🧪 Testing

### Run API Tests

```bash
# All API tests
pytest test_api.py -v

# Specific test class
pytest test_api.py::TestValidPredictions -v

# With coverage
pytest test_api.py --cov=api --cov-report=html
```

### Run Integration Tests

```bash
# All integration tests
pytest test_integration.py -v

# Specific test class
pytest test_integration.py::TestCanaryDeployment -v

# Run both with coverage
pytest test_*.py --cov=api --cov-report=html
```

### Test Categories

**API Tests** (`test_api.py`):
- ✅ Health checks and model info
- ✅ Valid predictions with various image sizes
- ✅ Error handling (invalid files, corrupt images, small images)
- ✅ A/B testing comparison
- ✅ Metrics and logging
- ✅ Performance benchmarks

**Integration Tests** (`test_integration.py`):
- ✅ Canary deployment traffic splitting
- ✅ Model version isolation
- ✅ Agreement rate calculation
- ✅ Latency tracking by version
- ✅ Error cascade prevention
- ✅ Stress testing (rapid requests)
- ✅ Data quality validation
- ✅ Performance SLA verification

## ☁️ Azure Deployment

### Prerequisites

1. **Azure Account**: Create at https://azure.microsoft.com/

2. **Azure CLI**: 
   ```bash
   # Windows (via PowerShell)
   choco install azure-cli
   
   # macOS
   brew install azure-cli
   
   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

3. **Login to Azure**:
   ```bash
   az login
   ```

### Automatic Setup (Recommended)

#### Windows (PowerShell)

```powershell
# Run the PowerShell setup script
.\azure_setup.ps1

# Or with custom parameters
.\azure_setup.ps1 -ResourceGroupName "my-adas-rg" -Location "westus"
```

#### Linux/macOS (Bash)

```bash
# Make executable
chmod +x azure_setup.sh

# Run the setup script
./azure_setup.sh

# Or with custom parameters
./azure_setup.sh adas-ml-rg eastus adas-ml-ws
```

### What the Setup Does

1. ✅ Creates Resource Group
2. ✅ Creates Storage Account (for model artifacts)
3. ✅ Creates Azure ML Workspace
4. ✅ Creates Container Registry
5. ✅ Creates Key Vault (for secrets)
6. ✅ Creates blob containers

### After Setup

The script outputs next steps. Follow them to:

#### 1. Build Docker Image for Azure

```bash
cd class2
az acr build --registry <your-registry-name> --image adas-api:latest .
```

#### 2. Deploy to Azure Container Instances

```bash
az container create \
  --resource-group adas-ml-rg \
  --name adas-api-container \
  --image <registry-url>/adas-api:latest \
  --registry-username <username> \
  --registry-password <password> \
  --ports 8000 8501 \
  --environment-variables \
    AZURE_ML_WORKSPACE=adas-ml-ws \
    AZURE_ML_RESOURCE_GROUP=adas-ml-rg \
  --cpu 2 --memory 4
```

#### 3. Monitor Deployment

```bash
# Check status
az container show \
  --resource-group adas-ml-rg \
  --name adas-api-container

# View logs
az container logs \
  --resource-group adas-ml-rg \
  --name adas-api-container
```

#### 4. Access Application

```bash
# Get container IP
az container show \
  --resource-group adas-ml-rg \
  --name adas-api-container \
  --query ipAddress.ip --output tsv

# Then access at http://<IP>:8000 and http://<IP>:8501
```

## 📊 API Endpoints

### Health & Info
```bash
# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/info
```

### Predictions
```bash
# Single prediction (canary routed)
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"

# Force specific model version
curl -X POST "http://localhost:8000/predict?model_version=v1" \
  -F "file=@image.jpg"

# A/B testing (both models)
curl -X POST http://localhost:8000/predict-both \
  -F "file=@image.jpg"
```

### Metrics & Monitoring
```bash
# Get metrics
curl http://localhost:8000/metrics

# Get detailed statistics
curl http://localhost:8000/stats

# Get prediction logs
curl http://localhost:8000/logs?limit=100
```

## 📁 Model Registry Integration

### Replace with Your Models

The API currently uses random weights for demo. To use real trained models:

#### 1. Save Your Model

From your Class 1 notebook:
```python
import torch
from api import BaselineModel, DropoutModel

# Load your trained model
model = BaselineModel(num_classes=7)
model.load_state_dict(torch.load('path/to/trained_model.pth'))

# Save for production
torch.save(model.state_dict(), 'models/v1.0/model.pth')
```

#### 2. Update API to Load Your Model

Edit `api.py`, find `load_models_from_azure()`:

```python
def load_models_from_azure():
    try:
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        
        # Load v1
        model_v1 = BaselineModel(NUM_CLASSES).to(DEVICE)
        v1_path = os.path.join(LOCAL_MODEL_PATH, 'v1.0', 'model.pth')
        if os.path.exists(v1_path):
            model_v1.load_state_dict(torch.load(v1_path, map_location=DEVICE))
        
        # Load v2
        model_v2 = DropoutModel(NUM_CLASSES).to(DEVICE)
        v2_path = os.path.join(LOCAL_MODEL_PATH, 'v2.0', 'model.pth')
        if os.path.exists(v2_path):
            model_v2.load_state_dict(torch.load(v2_path, map_location=DEVICE))
        
        model_v1.eval()
        model_v2.eval()
        
        return model_v1, model_v2
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise
```

#### 3. Upload to Azure ML (Advanced)

```bash
# Using Azure ML CLI
az ml model create \
  --name adas-model-v1 \
  --version 1 \
  --path models/v1.0/model.pth \
  --resource-group adas-ml-rg \
  --workspace-name adas-ml-ws
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `API_PORT`: API server port (default: 8000)
- `API_WORKERS`: Number of worker processes
- `CANARY_PERCENTAGE`: Traffic split between v1 and v2 (0-100)
- `AZURE_*`: Azure credentials
- `LOCAL_MODEL_PATH`: Where to load/store models

## 📈 Monitoring & Observability

### Built-in Metrics

The API automatically tracks:
- Request count by model version
- Latency (min, max, avg, std)
- Error rate and types
- Model agreement rate (for A/B tests)
- Confidence distribution
- Prediction distribution

### Access Metrics Dashboard

```bash
# Get current metrics
curl http://localhost:8000/metrics

# Get detailed statistics
curl http://localhost:8000/stats

# Export logs as CSV
curl http://localhost:8000/logs?limit=1000 | jq '.logs' > predictions.json
```

### Production Monitoring

For production, integrate with:
- **Azure Application Insights**
- **Prometheus + Grafana**
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **DataDog**

## 🔐 Security Best Practices

1. **API Authentication**: Add API key validation
2. **HTTPS**: Use SSL/TLS in production
3. **Rate Limiting**: Prevent abuse with rate limiters
4. **Input Validation**: All inputs are validated
5. **Logging**: All predictions are logged
6. **Secrets Management**: Use Azure Key Vault
7. **Private Registry**: Store Docker images privately

## 🐛 Troubleshooting

### API won't start
```bash
# Check if port is in use
netstat -tulpn | grep 8000

# Check logs
docker-compose logs api
```

### Docker build fails
```bash
# Clean up and rebuild
docker-compose down
docker system prune -a
docker-compose up --build
```

### Azure authentication errors
```bash
# Re-login to Azure
az logout
az login

# Check current subscription
az account show
```

### Test failures
```bash
# Run with verbose output
pytest test_api.py -vv -s

# Run single test
pytest test_api.py::TestHealthAndInfo::test_health_check_returns_200 -vv
```

## 📚 API Documentation

### Swagger/OpenAPI

Full interactive API documentation available at:
```
http://localhost:8000/docs
```

### Request/Response Examples

#### Prediction Request
```json
{
  "image": "<binary image data>",
  "model_version": "v1"  // Optional
}
```

#### Prediction Response
```json
{
  "image_id": "1704067200.123456",
  "prediction": "pothole",
  "confidence": 0.9847,
  "class_probabilities": {
    "animal": 0.0012,
    "name_board": 0.0003,
    "other_vehicle": 0.0008,
    "pedestrian": 0.0095,
    "pothole": 0.9847,
    "road_sign": 0.0021,
    "speed_breaker": 0.0014
  },
  "model_version": "v1.0",
  "latency_ms": 45.32
}
```

## 🧠 Understanding the Architecture

### Single Model Prediction (Canary)
```
User Request → API Router → Model Selection (70% v1, 30% v2)
              → Prediction → Log → Response
```

### A/B Testing (Comparison)
```
User Request → Get Image
            → v1 Prediction ──┐
            → v2 Prediction ──→ Compare → Log → Response
```

### Canary Deployment Strategy
```
Week 1-2:  10% v2, 90% v1  (Monitor)
Week 3-4:  25% v2, 75% v1  (Monitor)
Week 5-6:  50% v2, 50% v1  (Monitor)
Week 7-8: 100% v2           (Fully deployed)
```

## 📊 Example Workflows

### Workflow 1: Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@pothole.jpg" | jq .

# Response includes which model version was used
# Either v1.0 (70% chance) or v2.0 (30% chance)
```

### Workflow 2: A/B Testing
```bash
curl -X POST http://localhost:8000/predict-both \
  -F "file=@pothole.jpg" | jq .

# Response shows both v1 and v2 predictions
# Learn which model is better for your data
```

### Workflow 3: Monitor Metrics
```bash
while true; do
  curl http://localhost:8000/metrics | jq .
  sleep 5
done

# Watch agreement rate, latency, traffic distribution
```

## 📝 Class 1 & 2 Comparison

| Aspect | Class 1 (Notebooks) | Class 2 (Production) |
|--------|-------------------|-------------------|
| Format | Jupyter Notebooks | Python + Streamlit |
| Deployment | Local only | Docker + Azure |
| Model serving | Single model | Multiple versions |
| Testing | Manual | Automated (pytest) |
| Monitoring | Print statements | Metrics API |
| Production ready | ❌ | ✅ |

## 🎓 Learning Outcomes

After completing this project, you understand:

1. **API Design**: RESTful principles, error handling, versioning
2. **Containerization**: Docker, docker-compose, registry
3. **Cloud Deployment**: Azure CLI, Container Instances
4. **Testing**: Unit, integration, and stress testing
5. **Model Serving**: Canary deployments, A/B testing, version management
6. **Monitoring**: Logging, metrics, performance tracking
7. **Production Practices**: Security, scalability, reliability

## 📖 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure CLI Docs**: https://learn.microsoft.com/en-us/cli/azure/
- **Docker Docs**: https://docs.docker.com/
- **PyTorch Docs**: https://pytorch.org/docs/stable/

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review test files for examples
3. Check API documentation at `/docs`
4. Review logs: `docker-compose logs`

## 📄 License

This project is provided as educational material for BITS.

---

**Happy Deploying! 🚀**
