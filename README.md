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

## Requirements

- Python 3.10+
- Node.js (for serving the frontend)
- ~1GB disk space (for AI model weights)

## How to Run

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** First run will automatically download model weights:
> - OpenAI Whisper `base` model (~150MB)
> - WhisperX Wav2Vec2 English alignment model (~360MB)
> - Silero VAD model (~2MB, cached by PyTorch hub)

### 2. Start the Python Backend

```bash
cd backend
python server.py
```

The backend starts on **http://localhost:8000**. Wait for all three models to load:

```
Loading Whisper model (this may take a moment)...
Whisper model loaded successfully!
Loading WhisperX alignment model...
WhisperX alignment model loaded successfully!
Loading Silero VAD model...
Silero VAD model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Serve the Frontend

In a **separate terminal**, from the project root:

#### Option A: Python static server
```bash
python -m http.server 8080
```

#### Option B: Node.js / npx
```bash
npx http-server -p 8080
```

Then open **http://localhost:8080** in your browser.

### 4. Upload Audio

- Click **Upload Audio File** and select a `.wav`, `.mp3`, or `.m4a` file
- The frontend sends the file to the Python backend for processing
- Transcript with detected pauses appears automatically

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
├── index.html           # Main UI
├── app.js               # Frontend application logic
├── audio-analyzer.js    # Word-to-chunk assignment & transcript rendering
├── styles.css           # Styling
└── backend/
    ├── server.py        # FastAPI server (Whisper + WhisperX + Silero VAD)
    └── requirements.txt # Python dependencies
```
