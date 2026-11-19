FROM python:3.10-slim

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp/torch_cache \
    HF_HOME=/tmp/hf_cache \
    PORT=7860

# Install system dependencies (ffmpeg is mandatory for audio processing, git for model utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy dependency specifications first to leverage Docker layer caching
COPY --chown=user:user backend/requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files (backend + frontend)
COPY --chown=user:user . $HOME/app

# Pre-cache AI models (Whisper base, WhisperX alignment, Silero VAD) to eliminate runtime download delay
RUN python3 -c "import whisper; whisper.load_model('base'); import whisperx; whisperx.load_align_model(language_code='en', device='cpu'); import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)" || true

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI application via Uvicorn
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
