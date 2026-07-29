# EPICS Audio Acoustic Prototype

A minimalist, monochrome brutalist web application for micro-pause detection and speech transcription.

## Features
- **Acoustic Pause Detection:** Automatically detects millisecond speech pauses in uploaded audio or live microphone recordings.
- **Dynamic Silence Threshold:** Includes an adjustable slider to fine-tune pause detection sensitivity—ideal for noisy audio recordings.
- **Playback Synchronization:** Words and detected pauses are highlighted sequentially in real-time as the audio plays.
- **Auto-Activation:** Pause detection turns ON automatically once transcription is completed by the AI model.
- **100% Local Processing:** Runs entirely in your browser using Web Audio API and ONNX Runtime/Transformers.js for transcription (Whisper-Tiny).

## How to Run

To run the application locally, you just need to start a local static file server in the project directory.

### Method 1: Using Python (Recommended)
If you have Python installed, run:
```bash
python -m http.server 8080
```

### Method 2: Using Node.js / npx
If you have Node.js installed, run:
```bash
npx http-server -p 8080
```

Once the server is running, open your web browser and navigate to:
[http://localhost:8080](http://localhost:8080)
