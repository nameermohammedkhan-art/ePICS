---
title: ePICS Pause Detection & Speech Transcription
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ePICS — Pause Detection & Speech Transcription

A web application for detecting and analyzing rhetorical pauses in speech recordings, built for the EPICS project.

## Architecture

The application uses a **production-grade 3-stage pipeline** for precise pause placement:

```
Audio File
    │
    ├── Whisper ──────────────────► transcribed text + segment boundaries
    │       │
    │       └── WhisperX (Wav2Vec2) ──► word timestamps  [sole source]
    │
    └── Silero VAD ───────────────► speech segment boundaries → pauses [sole source]
                                              │
                          Merge: words → VAD chunks → render with pauses
```

| Component | Role |
|-----------|------|
| **OpenAI Whisper** | Transcription — converts audio to text with segment-level timing |
| **WhisperX (Wav2Vec2)** | Forced alignment — provides precise per-word start/end timestamps (±20–50ms accuracy) |
| **Silero VAD** | Voice Activity Detection — detects exact speech/silence boundaries to place pauses |

## Features

- **Rhetorical Pause Detection** — Detects meaningful pauses (400ms+) and ignores brief hesitations
- **Forced Alignment** — WhisperX replaces Whisper's native (drifting) word timestamps with phoneme-level accurate ones
- **VAD-Driven Pauses** — Pauses are derived exclusively from Silero VAD speech segment gaps, not from acoustic energy
- **Speech Chunk Rendering** — Words are grouped into speech blocks; pauses are rigidly inserted *between* blocks
- **Waveform Visualization** — Interactive waveform with pause markers
- **Playback Synchronization** — Words and pauses highlighted in real-time during audio playback

## Deploying to Hugging Face Spaces

This repository is ready to run as a Docker Space on Hugging Face:

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Enter Space Name: `ePICS`.
3. License: `mit`.
4. Select **Docker** as the Space SDK (Choose Blank).
5. Choose Free CPU (or T4 GPU for lightning-fast transcription).

### Step 2: Push to Hugging Face
You can push directly via Git:

```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/ePICS
git push space main
```

Or enable automated sync from GitHub:
1. Go to **Settings > Secrets and variables > Actions** in your GitHub repository.
2. Add a repository secret `HF_TOKEN` with a write-access Hugging Face token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3. Add a repository variable `HF_SPACE_REPO` set to `<YOUR_HF_USERNAME>/ePICS`.
4. Any push to `main` will automatically build and deploy your Space via the included GitHub Action.

---

## Running Locally

### Requirements

- Python 3.10+
- FFmpeg (required for audio decoding)
- ~1GB disk space (for AI model weights)

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** First run will automatically download model weights:
> - OpenAI Whisper `base` model (~150MB)
> - WhisperX Wav2Vec2 English alignment model (~360MB)
> - Silero VAD model (~2MB, cached by PyTorch hub)

### 2. Start the Application

You can start the unified server directly:

```bash
python -m backend.server
```

Then open **http://localhost:7860** in your browser. The backend automatically serves both the FastAPI endpoints and the interactive frontend.

Alternatively, if you run the frontend separately with Vite or a static HTTP server:
```bash
python backend/server.py     # Starts on port 8000
python -m http.server 8080   # Starts frontend on port 8080
```

### 3. Upload Audio

- Click **Choose Audio File** and select a `.wav`, `.mp3`, or `.m4a` file
- The application processes the file through Whisper, WhisperX alignment, and Silero VAD
- Transcript with detected pauses appears automatically with waveform playback synchronization

## Backend API

```
POST /transcribe
Content-Type: multipart/form-data
Body: file=<audio_file>

Response:
{
  "text": "full transcript string",
  "words": [{ "word": "Hi", "startMs": 240, "endMs": 461 }, ...],
  "speech_segments": [{ "startMs": 136, "endMs": 760 }, ...],
  "pauses": [{ "startMs": 760, "endMs": 1704, "durationMs": 944, "eventType": "VAD Pause" }, ...]
}
```

## Project Structure

```
epics/
├── Dockerfile                   # Hugging Face Spaces / Docker container definition
├── .dockerignore                # Excluded build files
├── .github/workflows/
│   └── sync_to_huggingface.yml  # Automated GitHub-to-Hugging Face Space sync
├── index.html                   # Main UI
├── app.js                       # Frontend application logic
├── audio-analyzer.js            # Word-to-chunk assignment & transcript rendering
├── styles.css                   # Styling
└── backend/
    ├── __init__.py
    ├── server.py                # FastAPI server (Whisper + WhisperX + Silero VAD + Static files)
    └── requirements.txt         # Python dependencies
```
