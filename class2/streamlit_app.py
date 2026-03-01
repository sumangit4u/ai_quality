"""
streamlit_app.py — Class 2 ADAS Demo
=====================================
Demonstrates a CNN deployed as a FastAPI, including:
  - Live image prediction via the /predict endpoint
  - A/B model comparison via /predict-both
  - Rate limiting: the API rejects requests after 5 in 60 seconds
  - Real-time metrics from /metrics

Usage:
    # Terminal 1 — start the API
    uvicorn api:app --reload --port 8000

    # Terminal 2 — start the UI
    streamlit run streamlit_app.py
"""

import time
import io

import requests
import streamlit as st
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL           = "http://localhost:8000"
RATE_LIMIT        = 5     # mirrors api.py RATE_LIMIT_REQUESTS
RATE_LIMIT_WINDOW = 60    # mirrors api.py RATE_LIMIT_WINDOW (seconds)

CLASS_ICONS = {
    "animal":        "🐄",
    "name_board":    "🪧",
    "vehicle":       "🚛",
    "pedestrian":    "🚶",
    "pothole":       "🕳️",
    "road_sign":     "🛑",
    "speed_breaker": "🔶",
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADAS Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "request_times"  not in st.session_state:
    st.session_state.request_times  = []   # track click timestamps for rate-limit display
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_comparison" not in st.session_state:
    st.session_state.last_comparison = None
if "rate_limited"    not in st.session_state:
    st.session_state.rate_limited    = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _api_ok() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _requests_this_window() -> int:
    now = time.time()
    return len([t for t in st.session_state.request_times if now - t < RATE_LIMIT_WINDOW])


def _record_request():
    st.session_state.request_times.append(time.time())


def _predict(image_bytes: bytes, filename: str, version: str | None):
    files  = {"file": (filename, image_bytes, "image/jpeg")}
    params = {"model_version": version} if version else {}
    return requests.post(f"{API_URL}/predict", files=files, params=params, timeout=15)


def _predict_both(image_bytes: bytes, filename: str):
    files = {"file": (filename, image_bytes, "image/jpeg")}
    return requests.post(f"{API_URL}/predict-both", files=files, timeout=15)


def _get_metrics():
    try:
        return requests.get(f"{API_URL}/metrics", timeout=5).json()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 ADAS Demo")
    st.caption("Class 2 — CNN deployed as FastAPI")

    # API health
    st.divider()
    if _api_ok():
        try:
            health = requests.get(f"{API_URL}/health", timeout=2).json()
            st.success("API Online ✅")
            st.write(f"Device: `{health.get('device', '?')}`")
            st.write(f"Version: `{health.get('model_version', '?')}`")
        except Exception:
            st.success("API Online ✅")
    else:
        st.error("API Offline ❌")
        st.code("uvicorn api:app --reload --port 8000", language="bash")

    # Rate limit tracker
    st.divider()
    st.subheader("Rate Limiter")
    used = _requests_this_window()
    remaining = max(0, RATE_LIMIT - used)

    col_a, col_b = st.columns(2)
    col_a.metric("Used",      used)
    col_b.metric("Remaining", remaining)

    bar_val = used / RATE_LIMIT
    if bar_val >= 1.0:
        st.error(f"⛔ Limit reached — wait {RATE_LIMIT_WINDOW}s")
    else:
        st.progress(bar_val)

    st.caption(f"Max **{RATE_LIMIT}** requests per **{RATE_LIMIT_WINDOW}s**")

    if remaining == 0:
        oldest = min(t for t in st.session_state.request_times
                     if time.time() - t < RATE_LIMIT_WINDOW)
        wait = int(RATE_LIMIT_WINDOW - (time.time() - oldest)) + 1
        st.warning(f"Resets in ~{wait}s")

    # Live metrics
    st.divider()
    st.subheader("Live API Metrics")
    m = _get_metrics()
    if m and m.get("total_requests", 0) > 0:
        st.metric("Total requests",  m["total_requests"])
        st.metric("Avg latency",     f"{m['avg_latency_ms']} ms")
        st.metric("Agreement rate",  f"{m['agreement_rate']} %")
        st.metric("v1 / v2 split",   f"{m['v1_requests']} / {m['v2_requests']}")
    else:
        st.info("No predictions yet.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🚗 ADAS Road Hazard Classifier")
st.markdown(
    "Upload a road image to classify it into one of 7 hazard categories. "
    "Switch between **Single Predict** and **A/B Compare** tabs to explore both endpoints."
)

tab_single, tab_ab = st.tabs(["🔍 Single Predict", "⚖️ A/B Compare"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single prediction with rate limiting demo
# ──────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Upload & Predict")
    st.markdown(
        "The API enforces a **rate limit of 5 requests per minute** per client IP. "
        "Click Predict more than 5 times quickly to see an HTTP 429 response."
    )

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)

        version_choice = st.radio(
            "Model version",
            ["Auto (canary split — 70% v1 / 30% v2)", "Force v1 (Baseline)", "Force v2 (Dropout)"],
            horizontal=True,
        )
        version_param = None
        if "v1" in version_choice:
            version_param = "v1"
        elif "v2" in version_choice:
            version_param = "v2"

        predict_btn = st.button("🔍 Predict", type="primary", disabled=not uploaded)

        if predict_btn and uploaded:
            used_now = _requests_this_window()

            if used_now >= RATE_LIMIT:
                # Client knows it will be rate-limited — show pre-emptive warning
                st.session_state.rate_limited = True
                oldest = min(t for t in st.session_state.request_times
                             if time.time() - t < RATE_LIMIT_WINDOW)
                wait = int(RATE_LIMIT_WINDOW - (time.time() - oldest)) + 1
                st.error(f"⛔ Rate limit reached ({used_now}/{RATE_LIMIT} requests used).")
                st.warning(f"Wait approximately **{wait} seconds** and try again.")
            else:
                st.session_state.rate_limited = False
                uploaded.seek(0)
                image_bytes = uploaded.read()
                _record_request()

                with st.spinner("Sending to API…"):
                    try:
                        resp = _predict(image_bytes, uploaded.name, version_param)
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach the API. Is `uvicorn api:app --reload --port 8000` running?")
                        resp = None

                if resp is not None:
                    if resp.status_code == 200:
                        st.session_state.last_prediction = resp.json()
                        st.success("Prediction received!")
                    elif resp.status_code == 429:
                        st.error("⛔ HTTP 429 — Rate Limit Exceeded")
                        st.code(resp.json().get("detail", "Too many requests"), language="text")
                        st.info(
                            "The **server** rejected this request because the rate limit was hit. "
                            "This is the `429 Too Many Requests` HTTP status in action."
                        )
                    elif resp.status_code == 400:
                        st.error(f"HTTP 400 — Bad Request: {resp.json().get('detail')}")
                    else:
                        st.error(f"Unexpected error {resp.status_code}: {resp.text[:200]}")

    with col_result:
        result = st.session_state.last_prediction
        if result:
            pred   = result["prediction"]
            icon   = CLASS_ICONS.get(pred, "❓")
            conf   = result["confidence"]
            ver    = result["model_version"]
            lat    = result["latency_ms"]
            probs  = result["class_probabilities"]

            st.markdown(f"## {icon} {pred.replace('_', ' ').title()}")
            st.progress(conf, text=f"Confidence: {conf * 100:.1f} %")

            # Probability bar chart using st.bar_chart
            st.markdown("**Class probabilities**")
            prob_display = {
                k.replace("_", " ").title(): round(v * 100, 2)
                for k, v in sorted(probs.items(), key=lambda x: -x[1])
            }
            st.bar_chart(prob_display, height=220)

            st.caption(f"Model: `{ver}` | Latency: `{lat} ms`")
        else:
            st.info("Upload an image and click **Predict** to see results here.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — A/B comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab_ab:
    st.markdown("### A/B Model Comparison")
    st.markdown(
        "Sends the **same image to both v1 (Baseline) and v2 (Dropout)** simultaneously. "
        "Use this to compare predictions and confidence before committing to a canary rollout."
    )

    ab_upload = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="ab_upload",
    )

    if ab_upload:
        st.image(ab_upload, caption="Uploaded image", use_column_width=True)

    ab_btn = st.button("⚖️ Compare Both Models", type="primary", disabled=not ab_upload)

    if ab_btn and ab_upload:
        ab_upload.seek(0)
        image_bytes = ab_upload.read()

        with st.spinner("Querying both models…"):
            try:
                resp = _predict_both(image_bytes, ab_upload.name)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API.")
                resp = None

        if resp is not None:
            if resp.status_code == 200:
                st.session_state.last_comparison = resp.json()
            else:
                st.error(f"Error {resp.status_code}: {resp.text[:200]}")

    comp = st.session_state.last_comparison
    if comp:
        agreed = comp["agreement"]
        agree_label = "✅ Models agree" if agreed else "⚠️ Models disagree"

        st.markdown(f"### {agree_label}")

        c1, c2 = st.columns(2)
        with c1:
            pred1 = comp["v1_prediction"]
            st.markdown(f"**v1 — Baseline**")
            st.markdown(f"## {CLASS_ICONS.get(pred1,'❓')} {pred1.replace('_',' ').title()}")
            st.progress(comp["v1_confidence"],
                        text=f"Confidence: {comp['v1_confidence']*100:.1f} %")
            st.caption(f"Latency: `{comp['v1_latency_ms']} ms`")

        with c2:
            pred2 = comp["v2_prediction"]
            st.markdown(f"**v2 — Dropout**")
            st.markdown(f"## {CLASS_ICONS.get(pred2,'❓')} {pred2.replace('_',' ').title()}")
            st.progress(comp["v2_confidence"],
                        text=f"Confidence: {comp['v2_confidence']*100:.1f} %")
            st.caption(f"Latency: `{comp['v2_latency_ms']} ms`")

        if not agreed:
            st.warning(
                f"v1 predicted **{pred1}** while v2 predicted **{pred2}**. "
                "This disagreement is worth investigating before a full rollout."
            )
    else:
        st.info("Upload an image and click **Compare Both Models** to see results here.")
