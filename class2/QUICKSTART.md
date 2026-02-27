# Class 2 — Instructor Demo Guide
## Deploy CNN as API & Integration Testing

> **Before class**: Run the one-time model export in class1 (Step 0 below).
> Everything else takes < 2 minutes to start.

---

## Step 0 — One-Time Model Export (from Class 1)

Run the last cell of `class1/Part_2_Overfitting_and_Generalization.ipynb`.
This trains three models and saves the two best to:

```
class2/models/
├── v1/
│   ├── model.pth        ← Baseline ResNet-18 (no regularisation, ~91% test acc)
│   └── metadata.json
└── v2/
    ├── model.pth        ← Dropout ResNet-18 (best model, ~97% test acc)
    └── metadata.json
```

You only need to do this **once**. The files persist for all future class2 sessions.

---

## Step 1 — Install & Start

**Install (do once):**
```bash
cd class2
pip install -r requirements-api.txt
pip install -r requirements-streamlit.txt
```

**Start the API (Terminal 1):**
```bash
cd class2
uvicorn api:app --reload --port 8000
```
You should see `Loaded v1 (Baseline) weights` and `Loaded v2 (Dropout) weights` in the log.
If the models aren't there yet, it falls back gracefully with a warning.

**Start the Streamlit UI (Terminal 2):**
```bash
cd class2
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

---

## How to Demo — Part by Part

---

### Part 1 (40 min) — Deploy CNN as API

**Goal**: Students understand what it means to wrap a model in a web endpoint.

**Demo script:**

1. Open `http://localhost:8000/docs` — show the auto-generated Swagger UI.
   - Point out `/health`, `/predict`, `/predict-both`, `/metrics`
   - "This is what FastAPI gives you for free."

2. Call `/health` in the browser. Show the JSON response.

3. Upload an image via Swagger `/predict` — walk through the request/response fields:
   - `prediction`, `confidence`, `class_probabilities`, `model_version`, `latency_ms`

4. Show input validation from the Swagger UI:
   - Upload a `.txt` file → `400 Bad Request`
   - "Why 400 and not 500? Because the error is the client's fault, not the server's."

5. Show the same from the Streamlit UI:
   - Upload image → click **Predict**
   - Show the probability bar chart
   - Toggle between "Auto (canary)", "Force v1", "Force v2"

**Key question to ask students:**
> "What would break in production if we skipped the input validation?"

---

### Part 2 (40 min) — Integration Testing

**Goal**: Students understand canary deployment, A/B testing, and latency measurement.

**Demo script:**

1. Switch to the **A/B Compare** tab in Streamlit.
   - Upload an image → click **Compare Both Models**
   - Show that v1 and v2 sometimes disagree
   - "When do we trust v2 enough to send it 100% of traffic?"

2. Call `/predict-both` a few times — watch the **Live API Metrics** sidebar update:
   - `agreement_rate` rises or falls
   - `v1 / v2 split` counts each call
   - `avg_latency` is slightly higher than single-predict — why? (Two models run sequentially)

3. Show `/logs` in the browser — scroll through raw prediction records.
   - "In production these go to a database, not memory."

4. **Rate limiting demo** — click **Predict** in the Single tab 6 times quickly:
   - Requests 1–5: succeed (200 OK)
   - Request 6: rejected — the sidebar shows "0 remaining", the UI shows the HTTP 429 message
   - "The server returned 429 Too Many Requests. This protects the API from being overwhelmed."
   - Wait 60 seconds — the counter resets automatically.

5. Open `api.py` and show the rate limiter code (~10 lines):
   - `check_rate_limit()` — sliding window, defaultdict of timestamps
   - "No Redis, no middleware library — just Python."

**Key question to ask students:**
> "If 100 users hit the API at the same time, what happens without rate limiting?"

---

### Part 3 (40 min) — Data Drift Simulation

**Goal**: Students see how input distribution shift degrades accuracy.

**Demo script (notebook-based):**

1. Open `Part_3_Data_Drift_Detection.ipynb` and run it.
2. Show the original vs brightness-shifted histograms side by side.
   - "The model was trained on images like the left one. The right one is what a dirty camera sees."
3. Show the KL divergence increasing as shift magnitude increases.
4. Show accuracy dropping from ~97% toward chance level (~14%) under heavy shift.
5. "In production, you'd set an alert threshold: if KL divergence > X, retrain."

**Key question to ask students:**
> "What real-world events cause data drift in an ADAS system?"
> (Answers: rain, night, lens fog, camera angle change, geographic change)

---

### Lab 2 (60 min) — Student Hands-On

Students open `Lab_2_System_Testing_Student.ipynb` and build:

| # | Deliverable | Hint cell points to… |
|---|-------------|----------------------|
| 1 | `/predict` endpoint (mini FastAPI) | Part 1 notebook |
| 2 | 8-test test suite | `test_api.py` patterns |
| 3 | Drift detection script | Part 3 notebook |
| 4 | v1 vs v2 comparison report | `/metrics` endpoint |

Release `Lab_2_System_Testing_Solution.ipynb` after submission.

---

## API Endpoints Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | none | Server status + device info |
| `/info` | GET | none | Model metadata |
| `/predict` | POST | none | Single prediction (canary routed) |
| `/predict?model_version=v1` | POST | none | Force v1 |
| `/predict?model_version=v2` | POST | none | Force v2 |
| `/predict-both` | POST | none | Both models, A/B response |
| `/metrics` | GET | none | Aggregated stats |
| `/stats` | GET | none | Detailed breakdown |
| `/logs?limit=50` | GET | none | Raw prediction log |
| `/docs` | GET | none | Swagger UI |

**Rate limit**: 5 requests per 60 seconds per IP on `/predict`.

---

## Quick Test Commands

```bash
# Health check
curl http://localhost:8000/health

# Predict (replace image.jpg with a real file)
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"

# Force v2
curl -X POST "http://localhost:8000/predict?model_version=v2" \
  -F "file=@image.jpg"

# A/B compare
curl -X POST http://localhost:8000/predict-both \
  -F "file=@image.jpg"

# Metrics
curl http://localhost:8000/metrics

# Trigger rate limit (run 6 times quickly)
for i in {1..6}; do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -X POST http://localhost:8000/predict \
    -F "file=@image.jpg"
done
```

---

## Troubleshooting

**"Loaded with random weights" in API log:**
The `class2/models/` folder is empty. Run the last cell of
`class1/Part_2_Overfitting_and_Generalization.ipynb` first.

**Port 8000 already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# macOS / Linux
lsof -ti:8000 | xargs kill
```

**Streamlit can't reach API:**
Make sure `uvicorn api:app --reload --port 8000` is running in a separate terminal.
The Streamlit app connects to `http://localhost:8000`.

**Rate limit resets:**
The sliding window is 60 seconds. Wait a minute and the counter resets automatically.
You can also restart the API server to clear all rate limit state.

---

## Learning Checklist

**Part 1**
- [ ] `/predict` returns `prediction`, `confidence`, `model_version`, `class_probabilities`
- [ ] Invalid file type → HTTP 400
- [ ] Corrupt image → HTTP 400
- [ ] Swagger UI open and working at `/docs`

**Part 2**
- [ ] Canary split visible in `/metrics` (`v1_requests` vs `v2_requests`)
- [ ] Rate limit triggered — HTTP 429 shown in Streamlit and API log
- [ ] A/B comparison shows when v1 and v2 disagree
- [ ] Students can read `/logs` and explain what each field means

**Part 3**
- [ ] Histogram comparison plotted for brightness and contrast shifts
- [ ] KL divergence increases with shift magnitude
- [ ] Accuracy drops are charted and discussed

**Lab 2**
- [ ] Student's mini API returns correct JSON
- [ ] At least 8 tests written and passing
- [ ] Drift detection script computes KL divergence
- [ ] Model comparison report table completed
