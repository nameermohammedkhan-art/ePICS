import os
import sys
import tempfile

# Compatibility fix for huggingface_hub >= 1.0 where HfFolder was removed
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return getattr(huggingface_hub, "get_token", lambda: None)()
            @staticmethod
            def save_token(token):
                if hasattr(huggingface_hub, "login"):
                    huggingface_hub.login(token=token)
            @staticmethod
            def delete_token():
                if hasattr(huggingface_hub, "logout"):
                    huggingface_hub.logout()
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# Compatibility fix for gradio_client JSON schema boolean bug with newer pydantic
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type
except Exception:
    pass

# Compatibility fix for Starlette >= 0.39 TemplateResponse signature change
try:
    import inspect
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse
    _sig = inspect.signature(_orig_template_response)
    _params = list(_sig.parameters.keys())
    if len(_params) >= 2 and _params[1] == "request":
        def _patched_template_response(self, *args, **kwargs):
            if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
                req = args[1].get("request")
                name = args[0]
                context = args[1]
                return _orig_template_response(self, req, name, context, **kwargs)
            return _orig_template_response(self, *args, **kwargs)
        starlette.templating.Jinja2Templates.TemplateResponse = _patched_template_response
except Exception:
    pass

import torch
import whisper
import whisperx
import pandas as pd
import gradio as gr

# Support Hugging Face ZeroGPU if available
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

# Configure cache directories for Hugging Face container permissions
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

print(f"[ePICS] Initializing (device: {get_device()})")

# Global model holders for lazy/cached loading
whisper_model = None
align_model = None
align_metadata = None
silero_model = None
silero_utils = None
loaded_device = None

def get_models():
    global whisper_model, align_model, align_metadata, silero_model, silero_utils, loaded_device
    device = get_device()
    if whisper_model is None or loaded_device != device:
        print(f"[ePICS] Loading Whisper 'base' model on {device}...")
        whisper_model = whisper.load_model("base", device=device)
        print("[ePICS] Whisper loaded successfully.")

        print(f"[ePICS] Loading WhisperX forced alignment model on {device}...")
        align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
        print("[ePICS] WhisperX alignment model loaded.")

        if silero_model is None:
            print("[ePICS] Loading Silero VAD model...")
            silero_model, silero_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            print("[ePICS] Silero VAD loaded.")
        loaded_device = device
    return whisper_model, align_model, align_metadata, silero_model, silero_utils, device

@gpu_decorator
def analyze_speech_and_pauses(audio_file_path, min_pause_ms=400):
    if not audio_file_path:
        return (
            "<p style='color: #ef4444; font-weight: 600;'>Please upload an audio file or record from microphone.</p>",
            None,
            None,
            {}
        )

    try:
        w_model, a_model, a_meta, v_model, v_utils, device = get_models()
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = v_utils

        # 1. Whisper Transcription
        whisper_result = w_model.transcribe(audio_file_path, word_timestamps=False)
        raw_text = whisper_result.get("text", "").strip()

        # 2. WhisperX Forced Alignment
        audio = whisperx.load_audio(audio_file_path)
        aligned = whisperx.align(
            whisper_result.get("segments", []),
            a_model,
            a_meta,
            audio,
            device,
            return_char_alignments=False
        )

        words_data = []
        for word in aligned.get("word_segments", []):
            if word.get("start") is None or word.get("end") is None:
                continue
            words_data.append({
                "word": word["word"].strip(),
                "startMs": int(word["start"] * 1000),
                "endMs": int(word["end"] * 1000)
            })

        # 3. Silero Voice Activity Detection (VAD)
        wav = read_audio(audio_file_path, sampling_rate=16000)
        audio_duration_sec = len(wav) / 16000.0
        total_duration_ms = int(audio_duration_sec * 1000)

        speech_timestamps = get_speech_timestamps(
            wav,
            v_model,
            sampling_rate=16000,
            min_silence_duration_ms=int(min_pause_ms),
            speech_pad_ms=120,
            min_speech_duration_ms=200
        )

        raw_segments = []
        for segment in speech_timestamps:
            raw_segments.append({
                "startMs": int(segment['start'] / 16.0),
                "endMs": int(segment['end'] / 16.0)
            })

        # Merge segments closer than min_pause_ms
        speech_chunks = []
        for seg in raw_segments:
            if not speech_chunks:
                speech_chunks.append(dict(seg))
                continue
            gap = seg["startMs"] - speech_chunks[-1]["endMs"]
            if gap < min_pause_ms:
                speech_chunks[-1]["endMs"] = seg["endMs"]
            else:
                speech_chunks.append(dict(seg))

        # Calculate rhetorical pauses from gaps between speech chunks
        pauses_data = []
        for i in range(len(speech_chunks) - 1):
            start_pause = speech_chunks[i]["endMs"]
            end_pause = speech_chunks[i+1]["startMs"]
            duration = end_pause - start_pause
            if duration >= min_pause_ms:
                pauses_data.append({
                    "pause_number": len(pauses_data) + 1,
                    "start_time_sec": round(start_pause / 1000.0, 2),
                    "end_time_sec": round(end_pause / 1000.0, 2),
                    "duration_ms": duration,
                    "start_ms": start_pause,
                    "end_ms": end_pause,
                })

        # 4. Group Words into Speech Chunks via Maximum Overlap
        for chunk in speech_chunks:
            chunk["words"] = []

        if speech_chunks:
            for w in words_data:
                best_idx = -1
                max_overlap = 0
                for idx, chunk in enumerate(speech_chunks):
                    overlap_start = max(w["startMs"], chunk["startMs"])
                    overlap_end = min(w["endMs"], chunk["endMs"])
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_idx = idx

                # Fallback to closest chunk if no overlap
                if best_idx == -1:
                    midpoint = (w["startMs"] + w["endMs"]) / 2.0
                    min_dist = float("inf")
                    for idx, chunk in enumerate(speech_chunks):
                        dist = min(abs(midpoint - chunk["startMs"]), abs(midpoint - chunk["endMs"]))
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx

                speech_chunks[best_idx]["words"].append(w)

        # 5. Build HTML Transcript with Styled Pause Badges
        html_parts = []
        for idx, chunk in enumerate(speech_chunks):
            chunk["words"].sort(key=lambda x: x["startMs"])
            chunk_text = " ".join(w["word"] for w in chunk["words"])
            if chunk_text:
                html_parts.append(f"<span style='font-size: 1.15rem; color: #1e293b; font-weight: 500;'>{chunk_text}</span>")

            # Check if there is a pause after this chunk
            if idx < len(speech_chunks) - 1:
                chunk_end = chunk["endMs"]
                next_start = speech_chunks[idx + 1]["startMs"]
                duration = next_start - chunk_end
                if duration >= min_pause_ms:
                    badge = (
                        f"<span style='display: inline-flex; align-items: center; gap: 4px; "
                        f"background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; "
                        f"padding: 2px 9px; margin: 0 6px; border-radius: 9999px; "
                        f"font-size: 0.85rem; font-weight: 700; font-family: monospace; "
                        f"vertical-align: middle; box-shadow: 0 1px 2px rgba(0,0,0,0.05);' "
                        f"title='Pause: {duration}ms (from {chunk_end}ms to {next_start}ms)'>"
                        f"⏸️ {duration} ms"
                        f"</span>"
                    )
                    html_parts.append(badge)

        annotated_transcript_html = (
            "<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; "
            "padding: 24px; line-height: 2.2; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>"
            + (" ".join(html_parts) if html_parts else "<p style='color: #64748b;'>No speech detected.</p>")
            + "</div>"
        )

        # 6. Summary Metrics Cards
        total_pause_time_ms = sum(p["duration_ms"] for p in pauses_data)
        total_speech_time_ms = total_duration_ms - total_pause_time_ms
        word_count = len(words_data)
        wpm = round((word_count / (audio_duration_sec / 60.0)), 1) if audio_duration_sec > 0 else 0

        metrics_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Total Duration</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 4px;">{round(audio_duration_sec, 2)}s</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Pauses Detected</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #7c3aed; margin-top: 4px;">{len(pauses_data)}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Pause Time</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #0284c7; margin-top: 4px;">{round(total_pause_time_ms / 1000.0, 2)}s</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Word Count</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 4px;">{word_count}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Speech Rate</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #059669; margin-top: 4px;">{wpm} <span style="font-size: 0.85rem; font-weight: 500;">WPM</span></div>
          </div>
        </div>
        """

        # 7. Pause Table DataFrame
        table_rows = []
        for p in pauses_data:
            table_rows.append({
                "Pause #": p["pause_number"],
                "Start Time (s)": p["start_time_sec"],
                "End Time (s)": p["end_time_sec"],
                "Duration (ms)": p["duration_ms"]
            })
        df_pauses = pd.DataFrame(table_rows) if table_rows else pd.DataFrame(columns=["Pause #", "Start Time (s)", "End Time (s)", "Duration (ms)"])

        # 8. Raw JSON Output
        raw_json = {
            "text": raw_text,
            "durationMs": total_duration_ms,
            "word_count": word_count,
            "words": words_data,
            "speech_segments": speech_chunks,
            "pauses": pauses_data
        }

        return metrics_html, annotated_transcript_html, df_pauses, raw_json

    except Exception as exc:
        import traceback
        traceback.print_exc()
        error_html = f"<div style='color: #dc2626; padding: 16px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;'><b>Error during analysis:</b> {str(exc)}</div>"
        return error_html, None, None, {"error": str(exc)}

# Gradio Interface
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.header-badge {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 8px;
}
"""

with gr.Blocks(css=custom_css, title="ePICS — Pause Detection & Speech Transcription") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 24px; padding-top: 10px;">
      <span class="header-badge">EPICS Speech Analytics</span>
      <h1 style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin: 4px 0 10px 0;">🎙️ ePICS Pause & Speech Analyzer</h1>
      <p style="color: #64748b; font-size: 1.05rem; max-width: 680px; margin: auto; line-height: 1.5;">
        Accurate rhetorical pause detection using OpenAI Whisper, WhisperX forced alignment, and Silero Voice Activity Detection (VAD).
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Recording or File",
                waveform_options=gr.WaveformOptions(waveform_color="#9333ea", show_recording_waveform=True)
            )

            pause_threshold = gr.Slider(
                minimum=150,
                maximum=1500,
                value=400,
                step=50,
                label="Minimum Pause Threshold (ms)",
                info="Gaps between speech longer than this threshold are marked as rhetorical pauses (default 400ms)."
            )

            btn_analyze = gr.Button("🔍 Analyze Speech & Detect Pauses", variant="primary", size="lg")

            # Include sample.wav if present in repository
            sample_path = "sample.wav"
            if os.path.exists(sample_path):
                gr.Examples(
                    examples=[[sample_path, 400]],
                    inputs=[audio_input, pause_threshold],
                    label="Sample Audio"
                )

        with gr.Column(scale=7):
            metrics_display = gr.HTML()
            transcript_display = gr.HTML(label="Transcript with Detected Pauses")

            with gr.Accordion("📋 Pause Timeline Table", open=True):
                pause_table = gr.Dataframe(
                    headers=["Pause #", "Start Time (s)", "End Time (s)", "Duration (ms)"],
                    datatype=["number", "number", "number", "number"],
                    label="Detected Pauses"
                )

            with gr.Accordion("🔍 Raw Timestamps & JSON", open=False):
                json_display = gr.JSON(label="Detailed Timestamp Payload")

    btn_analyze.click(
        fn=analyze_speech_and_pauses,
        inputs=[audio_input, pause_threshold],
        outputs=[metrics_display, transcript_display, pause_table, json_display]
    )

    gr.HTML("""
    <div style="margin-top: 32px; padding: 20px; background: #f8fafc; border-top: 1px solid #e2e8f0; border-radius: 12px; font-size: 0.9rem; color: #64748b; line-height: 1.6;">
      <b>Architecture:</b>
      Audio ➔ <b>OpenAI Whisper</b> (transcription) ➔ <b>WhisperX Wav2Vec2</b> (phoneme-level forced alignment) ➔ <b>Silero VAD</b> (boundary segmentation) ➔ <b>Speech Chunk Merging</b> (rhetorical pause placement).
    </div>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False)
