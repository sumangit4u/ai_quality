"""
api_secure.py — Secure ADAS API (Class 3 Reference Implementation)

Demonstrates production-grade security patterns:
  - JWT authentication via python-jose (HS256)
  - In-memory sliding-window rate limiting
  - Protected /predict endpoint
  - Proper error codes: 401 Unauthorized, 429 Too Many Requests

Usage (standalone):
    uvicorn api_secure:app --reload --port 8001

Test credentials:
    username: testuser
    password: testpassword
"""

import os
import io
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import bcrypt as _bcrypt
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------
# IMPORTANT: In production, load SECRET_KEY from environment variable:
#   SECRET_KEY = os.getenv("JWT_SECRET_KEY")
# Never hardcode secrets in source code.
SECRET_KEY = "adas-demo-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ---------------------------------------------------------------------------
# Password Hashing (using bcrypt directly — no passlib dependency)
# ---------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---------------------------------------------------------------------------
# Fake User Database (demo only)
# ---------------------------------------------------------------------------
# In production: use a real database (PostgreSQL, MongoDB, etc.)
# NEVER store plain-text passwords — always store hashed passwords.
FAKE_USERS_DB = {
    "testuser": {
        "username": "testuser",
        "hashed_password": _bcrypt.hashpw(b"testpassword", _bcrypt.gensalt()),
        "disabled": False,
    },
    "admin": {
        "username": "admin",
        "hashed_password": _bcrypt.hashpw(b"admin_secret", _bcrypt.gensalt()),
        "disabled": False,
    },
}

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    timestamp: str


# ---------------------------------------------------------------------------
# JWT Helper Functions
# ---------------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: bytes) -> bool:
    """Verify plain password against bcrypt hash (bytes)."""
    return _bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password)


def authenticate_user(username: str, password: str):
    """Return user dict if credentials valid, else False."""
    user = FAKE_USERS_DB.get(username)
    if not user:
        return False
    if user.get("disabled"):
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT with expiry."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    FastAPI dependency: validate Bearer token, return user dict.
    Raises 401 HTTPException on invalid/expired token.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = FAKE_USERS_DB.get(token_data.username)
    if user is None or user.get("disabled"):
        raise credentials_exception
    return user


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    In-memory sliding-window rate limiter.

    Tracks per-client request timestamps in a dict.
    No external dependencies (Redis, etc.) needed for demo.

    In production: use Redis-backed rate limiting for distributed deployments.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Return True if request is within limit, False if rate limited."""
        now = time.time()
        window_start = now - self.window_seconds
        # Prune timestamps outside the window
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > window_start
        ]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Return number of remaining requests in current window."""
        now = time.time()
        window_start = now - self.window_seconds
        active = [ts for ts in self._requests[client_id] if ts > window_start]
        return max(0, self.max_requests - len(active))

    def reset(self, client_id: str = None):
        """Reset limits for a specific client or all clients."""
        if client_id:
            self._requests.pop(client_id, None)
        else:
            self._requests.clear()


# Default: 10 requests per minute per client
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "animal", "name_board", "pedestrian", "pothole",
    "road_sign", "speed_breaker", "vehicle"
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_VERSION = "v3.0-secure"


class ADASModel(nn.Module):
    """ResNet-18 backbone for 7-class ADAS road hazard classification."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.resnet = resnet18(weights=None)  # Random weights for demo
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Load model (random weights for demo; replace with real checkpoint in production)
model = ADASModel(NUM_CLASSES).to(DEVICE)
model.eval()
logger.info(f"Model loaded on {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ADAS Secure API",
    description=(
        "Class 3: Production-grade ADAS inference API with JWT authentication "
        "and rate limiting. All /predict calls require a valid Bearer token."
    ),
    version="3.0.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Issue a JWT access token.

    Request (form-encoded):
        username=testuser&password=testpassword

    Response:
        {"access_token": "<jwt>", "token_type": "bearer"}
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Public health check — no authentication required."""
    return HealthResponse(
        status="healthy",
        version=MODEL_VERSION,
        device=str(DEVICE),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),  # JWT guard
):
    """
    Authenticated, rate-limited image classification endpoint.

    Requires:
        Authorization: Bearer <access_token>

    Returns HTTP 401 if token missing/invalid.
    Returns HTTP 429 if rate limit exceeded.
    Returns HTTP 400 for invalid image input.
    """
    # --- Rate limiting ---
    client_ip = (request.client.host if request.client else "unknown")
    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {rate_limiter.max_requests} requests "
                f"per {rate_limiter.window_seconds}s. Remaining: {remaining}"
            ),
        )

    start_time = time.time()

    # --- File validation ---
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {allowed_extensions}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # --- Image validation ---
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    if image.size[0] < 32 or image.size[1] < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small: {image.size}. Minimum: 32x32 pixels",
        )

    # --- Inference ---
    try:
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        latency_ms = (time.time() - start_time) * 1000
        class_probs = {
            CLASS_NAMES[i]: float(probs[0, i]) for i in range(NUM_CLASSES)
        }

        logger.info(
            f"Prediction: {CLASS_NAMES[predicted_idx.item()]} "
            f"({confidence.item():.3f}) | {latency_ms:.1f}ms | user={current_user['username']}"
        )

        return PredictionResponse(
            prediction=CLASS_NAMES[predicted_idx.item()],
            confidence=round(float(confidence.item()), 4),
            class_probabilities=class_probs,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
