import argparse
import torch
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Voice Focus Pipeline Demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    print("Loading Speech-to-Text Model (Whisper-Tiny)...")
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")

    print(f"Transcribing {args.audio}...")
    result = stt_pipe(args.audio)
    raw_transcript = result["text"].strip()
    
    print("\n--- VOICE FOCUS: OFF ---")
    print(f"Raw Transcript (includes background/other voices):\n{raw_transcript}\n")

    print("Loading LLM for Transcript Filtering (Flan-T5-Small)...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    prompt = (
        "The following is a messy transcript containing a primary speaker and background interruptions. "
        "Remove the background voices and output ONLY what the primary speaker said.\n\n"
        f"Transcript: {raw_transcript}\n\n"
        "Cleaned Transcript:"
    )

    print("Cleaning transcript (Voice Focus processing)...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=100)
    cleaned_transcript = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- VOICE FOCUS: ON ---")
    print(f"Cleaned Transcript (Primary speaker only):\n{cleaned_transcript}\n")

if __name__ == "__main__":
    main()
