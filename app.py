"""
Streamlit frontend dashboard for the Voice AI Evaluation Pipeline.

Provides four pages:
  1. Single Evaluation — upload & evaluate a single .wav file
  2. Batch Evaluation — run evaluation across a full dataset
  3. Report Viewer — load and download the latest evaluation report
  4. Settings & Models — check Ollama models and API health
"""

import json

import plotly.express as px
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice AI Eval",
    layout="wide",
    page_icon="🎙️",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Custom CSS — dark professional theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- global ---------- */
    html, body {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
    }

    /* ---------- header ---------- */
    .dashboard-header {
        background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 50%, #0e1117 100%);
        border: 1px solid #1e2a3a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .dashboard-header h1 {
        background: linear-gradient(90deg, #00ff88, #00cc6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .dashboard-header p {
        color: #8899aa;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ---------- metric cards ---------- */
    .metric-card {
        background: linear-gradient(145deg, #131820, #1a2233);
        border: 1px solid #1e2a3a;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.08);
    }
    .metric-card .label {
        color: #8899aa;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .green  { color: #00ff88; }
    .red    { color: #ff4b4b; }
    .yellow { color: #ffcc00; }
    .blue   { color: #4dabf7; }

    /* ---------- badge ---------- */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-green {
        background: rgba(0,255,136,0.12);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .badge-red {
        background: rgba(255,75,75,0.12);
        color: #ff4b4b;
        border: 1px solid rgba(255,75,75,0.3);
    }

    /* ---------- section divider ---------- */
    .section-title {
        color: #00ff88;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #1e2a3a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
        <h1>🎙️ Voice AI Evaluation Pipeline</h1>
        <p>Whisper · LangChain · Ollama — Speech-to-text evaluation & LLM benchmarking</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: build a metric card
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, color: str = "green") -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {color}">{value}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Helper: safe API call
# ---------------------------------------------------------------------------
def api_call(method: str, path: str, **kwargs):
    """Perform an HTTP request to the backend and return the JSON response."""
    url = f"{API_URL}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot reach the backend API. Is the FastAPI server running?")
        return None
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = str(e)
        st.error(f"❌ API error: {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🎙️ Single Evaluation", "🎤 Speak Mode", "📦 Batch Evaluation", "📊 Report Viewer", "⚙️ Settings & Models"],
)

# =====================================================================
# PAGE 1 — Single Evaluation
# =====================================================================
if page == "🎙️ Single Evaluation":
    st.markdown('<div class="section-title">Single Audio Evaluation</div>', unsafe_allow_html=True)

    col_upload, col_params = st.columns([2, 1])

    with col_upload:
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
        ground_truth = st.text_input("Ground-truth transcript", placeholder="e.g. What is the capital of France?")
        expected_answer = st.text_input("Expected answer", placeholder="e.g. Paris is the capital of France.")

    with col_params:
        whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=1)
        llm_model = st.text_input("Ollama model", value="llama3.2")

    if st.button("🚀 Run Evaluation", use_container_width=True):
        if not audio_file:
            st.warning("Please upload a .wav file first.")
        elif not ground_truth or not expected_answer:
            st.warning("Please fill in both the ground-truth transcript and expected answer.")
        else:
            with st.spinner("Evaluating… this may take a moment on first run while models load."):
                # Auto-detect MIME type from extension
                ext = audio_file.name.rsplit(".", 1)[-1].lower() if audio_file.name else "wav"
                mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
                            "ogg": "audio/ogg", "flac": "audio/flac", "webm": "audio/webm"}
                mime_type = mime_map.get(ext, "audio/wav")
                files = {"audio_file": (audio_file.name, audio_file.getvalue(), mime_type)}
                data = {
                    "ground_truth_transcript": ground_truth,
                    "expected_answer": expected_answer,
                    "whisper_model": whisper_model,
                    "llm_model": llm_model,
                }
                result = api_call("post", "/evaluate/single", files=files, data=data)

            if result:
                st.success("✅ Evaluation complete!")

                # --- Metric cards ---
                wer_val = result.get("wer", 0)
                sim_val = result.get("semantic_similarity", 0)
                lat_val = result.get("latency", 0)
                hall = result.get("hallucination", False)

                wer_color = "green" if wer_val < 0.1 else ("yellow" if wer_val <= 0.3 else "red")
                sim_color = "green" if sim_val > 0.8 else ("yellow" if sim_val >= 0.5 else "red")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(metric_card("WER", f"{wer_val:.4f}", wer_color), unsafe_allow_html=True)
                with c2:
                    st.markdown(metric_card("Semantic Similarity", f"{sim_val:.4f}", sim_color), unsafe_allow_html=True)
                with c3:
                    st.markdown(metric_card("Latency", f"{lat_val:.2f}s", "blue"), unsafe_allow_html=True)
                with c4:
                    if hall:
                        st.markdown(metric_card("Hallucination", "🔴 Detected", "red"), unsafe_allow_html=True)
                    else:
                        st.markdown(metric_card("Hallucination", "🟢 Clean", "green"), unsafe_allow_html=True)

                # --- Detail expanders ---
                with st.expander("📝 Transcription"):
                    st.write(result.get("transcription", ""))
                with st.expander("🤖 LLM Response"):
                    st.write(result.get("llm_response", ""))

# =====================================================================
# PAGE 2 — Speak Mode
# =====================================================================
elif page == "🎤 Speak Mode":
    st.markdown('<div class="section-title">🎤 Speak Mode — Record & Evaluate</div>', unsafe_allow_html=True)
    st.markdown(
        "Record from your microphone, then get instant transcription and an AI response. "
        "Optionally provide ground truth to see evaluation metrics."
    )

    # --- Mic recorder ---
    col_mic, col_params = st.columns([2, 1])

    with col_mic:
        st.markdown("#### 🎙️ Click to Record")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#00ff88",
            neutral_color="#8899aa",
            icon_size="3x",
            pause_threshold=3.0,
        )

    with col_params:
        whisper_model_s = st.selectbox(
            "Whisper model", ["tiny", "base", "small", "medium", "large"],
            index=1, key="speak_whisper",
        )
        llm_model_s = st.text_input("Ollama model", value="llama3.2", key="speak_llm")

    # Show playback if audio was recorded
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    # --- Optional ground truth ---
    with st.expander("📝 Optional: Provide Ground Truth for Evaluation Metrics"):
        gt_speak = st.text_input(
            "Ground-truth transcript",
            placeholder="e.g. What is the capital of France?",
            key="speak_gt",
        )
        ea_speak = st.text_input(
            "Expected answer",
            placeholder="e.g. Paris is the capital of France.",
            key="speak_ea",
        )

    # --- Evaluate button ---
    if st.button("🚀 Evaluate Recording", use_container_width=True, key="speak_eval"):
        if not audio_bytes:
            st.warning("Please record some audio first by clicking the mic icon above.")
        else:
            with st.spinner("Processing your recording… this may take a moment."):
                files = {"audio_file": ("mic_recording.wav", audio_bytes, "audio/wav")}
                form_data = {
                    "whisper_model": whisper_model_s,
                    "llm_model": llm_model_s,
                }
                if gt_speak:
                    form_data["ground_truth_transcript"] = gt_speak
                if ea_speak:
                    form_data["expected_answer"] = ea_speak

                result = api_call("post", "/evaluate/speak", files=files, data=form_data)

            if result:
                is_full = result.get("mode") == "full_evaluation"
                st.success("✅ Processing complete!")

                # --- Always show: Transcription & LLM Response ---
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**📝 Transcription**")
                    st.info(result.get("transcription", ""))
                with r2:
                    st.markdown("**🤖 AI Response**")
                    st.success(result.get("llm_response", ""))

                # --- Latency card ---
                lat_val = result.get("latency", 0)
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    st.markdown(metric_card("Latency", f"{lat_val:.2f}s", "blue"), unsafe_allow_html=True)

                # --- Metric cards (only if full evaluation) ---
                if is_full:
                    wer_val = result.get("wer", 0)
                    sim_val = result.get("semantic_similarity", 0)
                    hall = result.get("hallucination", False)

                    wer_color = "green" if wer_val < 0.1 else ("yellow" if wer_val <= 0.3 else "red")
                    sim_color = "green" if sim_val > 0.8 else ("yellow" if sim_val >= 0.5 else "red")

                    with lc2:
                        st.markdown(metric_card("WER", f"{wer_val:.4f}", wer_color), unsafe_allow_html=True)
                    with lc3:
                        st.markdown(metric_card("Semantic Similarity", f"{sim_val:.4f}", sim_color), unsafe_allow_html=True)
                    with lc4:
                        if hall:
                            st.markdown(metric_card("Hallucination", "🔴 Detected", "red"), unsafe_allow_html=True)
                        else:
                            st.markdown(metric_card("Hallucination", "🟢 Clean", "green"), unsafe_allow_html=True)
                else:
                    with lc2:
                        st.markdown(
                            metric_card("Metrics", "N/A", "yellow"),
                            unsafe_allow_html=True,
                        )
                    st.caption("💡 Provide ground truth and expected answer above to see WER, similarity, and hallucination metrics.")

# =====================================================================
# PAGE 3 — Batch Evaluation
# =====================================================================
elif page == "📦 Batch Evaluation":
    st.markdown('<div class="section-title">Batch Dataset Evaluation</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        audio_dir = st.text_input("Audio directory", value="dataset/audio")
        whisper_model_b = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=1, key="batch_whisper")
    with c2:
        gt_path = st.text_input("Ground-truth JSON path", value="dataset/ground_truth.json")
        llm_model_b = st.text_input("Ollama model", value="llama3.2", key="batch_llm")

    if st.button("📦 Run Batch Evaluation", use_container_width=True):
        with st.spinner("Running batch evaluation… this may take a while."):
            payload = {
                "audio_dir": audio_dir,
                "ground_truth_path": gt_path,
                "whisper_model": whisper_model_b,
                "llm_model": llm_model_b,
            }
            result = api_call("post", "/evaluate/batch", json=payload)

        if result:
            st.success("✅ Batch evaluation complete!")

            summary = result.pop("__summary__", None)

            # --- Summary cards ---
            if summary:
                st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(metric_card("Avg WER", f"{summary['avg_wer']:.4f}",
                                            "green" if summary["avg_wer"] < 0.1 else "red"), unsafe_allow_html=True)
                with s2:
                    st.markdown(metric_card("Avg Similarity", f"{summary['avg_semantic_similarity']:.4f}",
                                            "green" if summary["avg_semantic_similarity"] > 0.8 else "yellow"), unsafe_allow_html=True)
                with s3:
                    st.markdown(metric_card("Avg Latency", f"{summary['avg_latency']:.2f}s", "blue"), unsafe_allow_html=True)
                with s4:
                    rate = summary.get("hallucination_rate", 0)
                    st.markdown(metric_card("Hallucination Rate", f"{rate:.0%}",
                                            "green" if rate == 0 else "red"), unsafe_allow_html=True)

            # --- Per-file results ---
            if result:
                st.markdown('<div class="section-title">Per-file Results</div>', unsafe_allow_html=True)

                for fname, data in result.items():
                    with st.expander(f"📄 {fname}"):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            st.metric("WER", f"{data['wer']:.4f}")
                        with mc2:
                            st.metric("Similarity", f"{data['semantic_similarity']:.4f}")
                        with mc3:
                            st.metric("Latency", f"{data['latency']:.2f}s")
                        with mc4:
                            st.metric("Hallucination", "🔴 Yes" if data["hallucination"] else "🟢 No")
                        st.write("**Transcription:**", data.get("transcription", ""))
                        st.write("**LLM Response:**", data.get("llm_response", ""))

                # --- Charts ---
                st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

                chart_tab1, chart_tab2 = st.tabs(["Semantic Similarity", "WER"])

                filenames = list(result.keys())
                sim_values = [result[f]["semantic_similarity"] for f in filenames]
                wer_values = [result[f]["wer"] for f in filenames]

                with chart_tab1:
                    fig_sim = px.bar(
                        x=filenames, y=sim_values,
                        labels={"x": "File", "y": "Semantic Similarity"},
                        title="Semantic Similarity per File",
                        color=sim_values,
                        color_continuous_scale=["#ff4b4b", "#ffcc00", "#00ff88"],
                        range_color=[0, 1],
                    )
                    fig_sim.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="#c0c0c0",
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

                with chart_tab2:
                    fig_wer = px.bar(
                        x=filenames, y=wer_values,
                        labels={"x": "File", "y": "WER"},
                        title="Word Error Rate per File",
                        color=wer_values,
                        color_continuous_scale=["#00ff88", "#ffcc00", "#ff4b4b"],
                        range_color=[0, 1],
                    )
                    fig_wer.update_layout(
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="#c0c0c0",
                    )
                    st.plotly_chart(fig_wer, use_container_width=True)

# =====================================================================
# PAGE 3 — Report Viewer
# =====================================================================
elif page == "📊 Report Viewer":
    st.markdown('<div class="section-title">Evaluation Report Viewer</div>', unsafe_allow_html=True)

    if st.button("📂 Load Latest Report", use_container_width=True):
        with st.spinner("Loading report…"):
            report = api_call("get", "/report")

        if report:
            st.success("✅ Report loaded!")

            summary = report.get("__summary__")
            if summary:
                st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Total Samples", summary["total_samples"])
                with s2:
                    st.metric("Avg WER", f"{summary['avg_wer']:.4f}")
                with s3:
                    st.metric("Avg Similarity", f"{summary['avg_semantic_similarity']:.4f}")
                with s4:
                    st.metric("Hallucination Rate", f"{summary.get('hallucination_rate', 0):.0%}")

            st.markdown('<div class="section-title">Full Report JSON</div>', unsafe_allow_html=True)
            st.json(report)

            st.download_button(
                label="⬇️ Download Report",
                data=json.dumps(report, indent=2),
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True,
            )

# =====================================================================
# PAGE 4 — Settings & Models
# =====================================================================
elif page == "⚙️ Settings & Models":
    st.markdown('<div class="section-title">Settings & Models</div>', unsafe_allow_html=True)

    # --- API Health ---
    st.subheader("🏥 API Health")
    if st.button("Check API Health", use_container_width=True):
        data = api_call("get", "/")
        if data:
            st.success(f"✅ {data.get('message', 'API is running')}")

    st.divider()

    # --- Ollama Models ---
    st.subheader("🤖 Available Ollama Models")
    if st.button("Refresh Models", use_container_width=True):
        data = api_call("get", "/models")
        if data:
            models = data.get("models", [])
            if models:
                for m in models:
                    st.markdown(f'<span class="badge badge-green">{m}</span>', unsafe_allow_html=True)
            else:
                st.info("No models found. Pull a model with `ollama pull <model>`.")

    st.divider()

    # --- Config info ---
    st.subheader("📋 Configuration")
    config_data = {
        "API URL": API_URL,
        "Default Whisper Model": "base",
        "Default LLM Model": "llama3.2",
        "Ollama Server": "http://localhost:11434",
        "Reports Directory": "reports/",
    }
    for key, val in config_data.items():
        st.text(f"{key}: {val}")
