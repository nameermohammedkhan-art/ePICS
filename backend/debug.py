import whisper
import sys

model = whisper.load_model('base')
result = model.transcribe('C:\\Users\\user\\Downloads\\epics\\robot_sample.wav', word_timestamps=True)
words = []
for segment in result.get('segments', []):
    words.extend(segment.get('words', []))
for w in words:
    print(f"{w['word'].strip()}: {w['start']:.3f} - {w['end']:.3f}")
