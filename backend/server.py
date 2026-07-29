import os
import tempfile
import whisper
import whisperx
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

print("Loading Whisper model (this may take a moment)...")
# Load the 'base' model via openai-whisper for transcription
model = whisper.load_model("base")
print("Whisper model loaded successfully!")

print("Loading WhisperX alignment model...")
# Load the Wav2Vec2 forced alignment model for English
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
print("WhisperX alignment model loaded successfully!")

print("Loading Silero VAD model...")
silero_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("Silero VAD model loaded successfully!")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        file_size = os.path.getsize(temp_file_path)
        print("Processing file:", temp_file_path)
        print(f"File Name: {file.filename}, Size: {file_size} bytes")

        # BRANCH 1: Whisper — transcription only (text + segment boundaries, no word timestamps)
        print("Whisper input:", temp_file_path)
        print(f"Transcribing {file.filename} with Whisper...")
        whisper_result = model.transcribe(temp_file_path, word_timestamps=False)

        # BRANCH 2: WhisperX forced alignment — SOLE source of word start/end times
        # Whisper's native DTW timestamps are intentionally NOT used.
        # WhisperX aligns every word to exact phoneme boundaries via Wav2Vec2.
        print("Running WhisperX forced alignment for precise word boundaries...")
        audio = whisperx.load_audio(temp_file_path)
        aligned = whisperx.align(
            whisper_result["segments"],
            align_model,
            align_metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )

        # Extract from word_segments — this is the top-level flat list WhisperX provides
        words_data = []
        print("\n--- WhisperX aligned word timestamps ---")
        print(f"{'Word':<20} {'Start':>8} {'End':>8}")
        print("-" * 40)
        for word in aligned.get("word_segments", []):
            # Guard: WhisperX occasionally fails to align a word — skip nulls
            if word.get("start") is None or word.get("end") is None:
                print(f"  [skipped - no alignment]: {word.get('word', '?')}")
                continue
            start_ms = int(word["start"] * 1000)
            end_ms = int(word["end"] * 1000)
            text = word["word"].strip()
            print(f"{text:<20} {start_ms:>8} {end_ms:>8}")
            words_data.append({
                "word": text,
                "startMs": start_ms,
                "endMs": end_ms
            })

        print(f"\nAligned {len(words_data)} words total.")

        # Step 3: Silero VAD for speech segment boundaries
        print("Silero input:", temp_file_path)
        print(f"Detecting speech boundaries with Silero VAD...")
        wav = read_audio(temp_file_path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav,
            silero_model,
            sampling_rate=16000,
            min_silence_duration_ms=450,
            speech_pad_ms=120,
            min_speech_duration_ms=200
        )

        raw_segments = []
        for segment in speech_timestamps:
            start_ms = int(segment['start'] / 16.0)
            end_ms = int(segment['end'] / 16.0)
            raw_segments.append({
                "startMs": start_ms,
                "endMs": end_ms
            })

        # Post-process: merge speech segments separated by gaps shorter than 400ms
        speech_segments = []
        for seg in raw_segments:
            if not speech_segments:
                speech_segments.append(seg)
                continue

            gap = seg["startMs"] - speech_segments[-1]["endMs"]
            if gap < 400:
                speech_segments[-1]["endMs"] = seg["endMs"]
            else:
                speech_segments.append(seg)

        # Pauses are derived ONLY from gaps between VAD speech segments
        pauses_data = []
        for i in range(len(speech_segments) - 1):
            end_ms = speech_segments[i]["endMs"]
            start_ms = speech_segments[i+1]["startMs"]
            duration = start_ms - end_ms
            if duration > 0:
                pauses_data.append({
                    "startMs": end_ms,
                    "endMs": start_ms,
                    "durationMs": duration,
                    "type": "silence",
                    "eventType": "VAD Pause"
                })

        print("\n--- Silero VAD Speech Segments ---")
        for i, seg in enumerate(speech_segments):
            print(f"  Segment {i+1}: {seg['startMs']}ms -> {seg['endMs']}ms")
        print("\n--- Pauses (from VAD gaps) ---")
        for p in pauses_data:
            print(f"  Pause: {p['startMs']}ms -> {p['endMs']}ms ({p['durationMs']}ms)")

        return {
            "text": whisper_result["text"].strip(),
            "words": words_data,
            "speech_segments": speech_segments,
            "pauses": pauses_data
        }
    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
