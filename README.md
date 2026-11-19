---
title: ePICS Pause Detection & Speech Transcription
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
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
- **Waveform & Audio Controls** — Interactive audio recorder and playback with pause badges
- **Summary Metrics** — Speech rate (WPM), pause count, pause ratio, and detailed pause table

---

## Deploying to Hugging Face Spaces

This repository is ready to run as a **Gradio Space with Free ZeroGPU** or as a **Docker Space**:

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Space Name: `ePICS` (or your choice).
3. License: `mit`.
4. Space SDK: Select **Gradio** (Recommended for free ZeroGPU) or **Docker**.
5. Hardware: **ZeroGPU** (free) or standard CPU.

### Step 2: Push your Code to Hugging Face

**Option A (Direct Git Push):**
```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/ePICS
git push space main
```

**Option B (Automated GitHub Actions Sync):**
1. In your GitHub repository, navigate to **Settings > Secrets and variables > Actions**.
2. Add a Secret `HF_TOKEN` with your write token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Add a Variable `HF_SPACE_REPO` set to `<YOUR_HF_USERNAME>/ePICS`.
4. Any push to `main` automatically deploys to Hugging Face!

---

## Running Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Gradio Demo

```bash
python app.py
```
Open **http://localhost:7860** in your browser.

### 3. Run FastAPI + Custom Frontend Prototype

```bash
python -m backend.server
```
Or with custom port:
```bash
python backend/server.py
```

---

## Project Structure

```
epics/
├── app.py                       # Hugging Face Space Gradio application (ZeroGPU supported)
├── packages.txt                 # Debian system packages for Hugging Face (ffmpeg)
├── requirements.txt             # Python dependencies for Hugging Face
├── Dockerfile                   # Docker container definition (for Docker Spaces)
├── .dockerignore                # Container build ignore rules
├── .github/workflows/
│   └── sync_to_huggingface.yml  # Automated GitHub-to-Hugging Face Space sync
├── index.html                   # Custom HTML prototype UI
├── app.js                       # Custom prototype frontend logic
├── audio-analyzer.js            # In-browser chunk assignment and rendering
├── styles.css                   # Custom styles
├── sample.wav                   # Demo audio file
└── backend/
    ├── __init__.py
    ├── server.py                # FastAPI server (Whisper + WhisperX + Silero VAD)
    └── requirements.txt         # Backend Python dependencies
```
