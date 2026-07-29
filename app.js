import { AudioAnalyzerEngine } from './audio-analyzer.js';
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

env.allowLocalModels = false;

class App {
  constructor() {
    this.engine = new AudioAnalyzerEngine();

    // Application State
    this.audioBuffer = null;
    this.analysisResult = null;
    this.wordsWithTimestamps = [];
    this.audioSource = null;
    this.isPlaying = false;
    this.startTime = 0;
    this.pausedAt = 0;
    this.animFrameId = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;

    // Speech Recognition State
    this.speechRecognizer = null;
    this.isTranscribing = false;
    this.autoRecognizedText = "";

    // Pause Display Threshold Filter (default: 100ms)
    this.displayMinPauseMs = 100;

    // DOM Elements
    this.dropzone = document.getElementById('dropzone');
    this.fileInput = document.getElementById('audio-file-input');
    this.btnBrowseFile = document.getElementById('btn-browse-file');
    this.btnLoadDemo = document.getElementById('btn-load-demo');
    this.btnRecordMic = document.getElementById('btn-record-mic');

    this.playerCard = document.getElementById('player-card');
    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.iconPlay = document.getElementById('icon-play');
    this.iconPause = document.getElementById('icon-pause');
    this.playTime = document.getElementById('play-time');
    this.currentFilename = document.getElementById('current-filename');
    this.currentDuration = document.getElementById('current-duration');

    this.canvas = document.getElementById('waveform-canvas');
    this.ctx = this.canvas.getContext('2d');

    this.transcriptDisplay = document.getElementById('transcript-box');
    this.showPauses = false;
    
    // LLM state
    this.llmGenerator = null;
    this.isPipelineRunning = false;

    this.initEventListeners();
    this.initSpeechRecognition();
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  initSpeechRecognition() {
    this.transcriber = null;
  }

  initEventListeners() {
    if (this.btnBrowseFile) {
      this.btnBrowseFile.addEventListener('click', (e) => {
        e.stopPropagation();
        this.fileInput.click();
      });
    }

    this.dropzone.addEventListener('click', (e) => {
      if (e.target.closest('#btn-record-mic')) return;
      this.fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove('dragover');
      }, false);
    });

    this.dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        this.handleFile(files[0]);
      }
    });

    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.handleFile(e.target.files[0]);
      }
    });

    // Buttons
    this.btnLoadDemo.addEventListener('click', () => this.loadDemoRajuAudio());
    this.btnRecordMic.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMicRecording();
    });

    this.btnPlayPause.addEventListener('click', () => this.togglePlayPause());
    this.canvas.addEventListener('click', (e) => this.seekFromCanvas(e));

    const vfOffBtn = document.getElementById('vf-off');
    const vfOnBtn = document.getElementById('vf-on');

    if (vfOffBtn && vfOnBtn) {
      vfOnBtn.addEventListener('click', () => {
        vfOnBtn.style.background = 'var(--text-main)';
        vfOnBtn.style.color = 'var(--bg-color)';
        vfOffBtn.style.background = 'var(--bg-color)';
        vfOffBtn.style.color = 'var(--text-main)';
        
        this.showPauses = true;
        if (this.analysisResult) this.renderAnnotatedTranscript();
      });

      vfOffBtn.addEventListener('click', () => {
        vfOffBtn.style.background = 'var(--text-main)';
        vfOffBtn.style.color = 'var(--bg-color)';
        vfOnBtn.style.background = 'var(--bg-color)';
        vfOnBtn.style.color = 'var(--text-main)';
        
        this.showPauses = false;
        if (this.analysisResult) this.renderAnnotatedTranscript();
      });
    }
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = 120;
    if (this.analysisResult) {
      this.renderWaveform();
    }
  }

  async handleFile(file) {
    if (!file) return;
    try {
      this.currentFilename.textContent = `Analyzing ${file.name}...`;
      const arrayBuffer = await file.arrayBuffer();
      
      const ctx = this.engine.getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      this.audioBuffer = await this.engine.decodeAudioData(arrayBuffer);
      this.currentFilename.textContent = file.name;
      this.wordsWithTimestamps = [];
      this.autoRecognizedText = "";
      
      if (this.customTranscriptInput && !this.customTranscriptInput.value.trim()) {
        // Only clear if empty or just whitespace
        this.customTranscriptInput.value = "";
      }

      this.processAudioBuffer();
      
      if (!this.customTranscriptInput || !this.customTranscriptInput.value.trim()) {
        this.autoTranscribeAudio();
      }
    } catch (err) {
      console.error("Error decoding audio file:", err);
      alert(`Unable to process audio file "${file.name}". Error: ${err.message}\n\nPlease try standard WAV, MP3, M4A, or WebM format.`);
      this.currentFilename.textContent = "Error loading audio file";
    }
  }

  async loadDemoRajuAudio() {
    const ctx = this.engine.getAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    const sampleRate = ctx.sampleRate;

    // Create synthetic audio buffer matching prompt details:
    // "My name is Raju (pause detected: 17 ms). I am a football (pause detected: 13 ms) player."
    const totalDurationSec = 3.5;
    const length = Math.floor(sampleRate * totalDurationSec);
    const audioBuffer = ctx.createBuffer(1, length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);

    const msToSample = (ms) => Math.floor((ms / 1000) * sampleRate);

    // Generate speech tones
    const addSpeechSegment = (startMs, endMs, baseFreq) => {
      const startS = msToSample(startMs);
      const endS = msToSample(endMs);
      for (let i = startS; i < endS && i < length; i++) {
        const t = (i - startS) / sampleRate;
        const envelope = Math.sin(Math.PI * (i - startS) / (endS - startS));
        const wave = 0.3 * Math.sin(2 * Math.PI * baseFreq * t) +
                     0.15 * Math.sin(2 * Math.PI * (baseFreq * 2) * t) +
                     0.08 * Math.sin(2 * Math.PI * (baseFreq * 3) * t);
        channelData[i] = wave * envelope;
      }
    };

    // Speech 1: "My name is Raju" (0 to 1200ms)
    addSpeechSegment(50, 300, 180);   // My
    addSpeechSegment(340, 600, 210);  // name
    addSpeechSegment(630, 800, 195);  // is
    addSpeechSegment(840, 1200, 175); // Raju

    // Pause 1: 1200ms to 1217ms (17ms pause)

    // Speech 2: "I am a football" (1217ms to 2400ms)
    addSpeechSegment(1217, 1400, 200); // I
    addSpeechSegment(1440, 1650, 190); // am
    addSpeechSegment(1680, 1780, 185); // a
    addSpeechSegment(1820, 2400, 170); // football

    // Pause 2: 2400ms to 2413ms (13ms pause)

    // Speech 3: "player." (2413ms to 3100ms)
    addSpeechSegment(2413, 3100, 180); // player

    // Reference Words
    this.wordsWithTimestamps = [
      { word: "My", startMs: 50, endMs: 300 },
      { word: "name", startMs: 340, endMs: 600 },
      { word: "is", startMs: 630, endMs: 800 },
      { word: "Raju", startMs: 840, endMs: 1200 },
      { word: "I", startMs: 1217, endMs: 1400 },
      { word: "am", startMs: 1440, endMs: 1650 },
      { word: "a", startMs: 1680, endMs: 1780 },
      { word: "football", startMs: 1820, endMs: 2400 },
      { word: "player.", startMs: 2413, endMs: 3100 }
    ];

    this.audioBuffer = audioBuffer;
    this.currentFilename.textContent = "raju_sample_audio.wav (Demo)";
    
    // Set slider to 10ms for Raju demo so 17ms and 13ms pauses display as specified
    this.displayMinPauseMs = 10;
    if (this.sliderMinPause) this.sliderMinPause.value = 10;
    if (this.valMinPauseLabel) this.valMinPauseLabel.textContent = "10 ms";

    if (this.customTranscriptInput) {
      this.customTranscriptInput.value = "My name is Raju. I am a football player.";
    }

    this.processAudioBuffer();
  }

  async autoTranscribeAudio() {
    if (!this.audioBuffer) return;

    if (this.isTranscribing) return;
    this.isTranscribing = true;

    if (this.transcriptDisplay) {
      this.transcriptDisplay.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9em;">⏳ Loading AI transcription model... (This may take a minute on first run)</span>`;
    }

    try {
      const offlineCtx = new window.OfflineAudioContext(1, this.audioBuffer.duration * 16000, 16000);
      const source = offlineCtx.createBufferSource();
      source.buffer = this.audioBuffer;
      source.connect(offlineCtx.destination);
      source.start();
      const resampledBuffer = await offlineCtx.startRendering();
      const audioData = resampledBuffer.getChannelData(0);

      if (!this.transcriber) {
        this.transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
      }

      if (this.transcriptDisplay) {
        this.transcriptDisplay.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9em;">🎙️ Transcribing Audio...</span>`;
      }

      const output = await this.transcriber(audioData, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: 'word'
      });

      if (output.chunks) {
        this.wordsWithTimestamps = output.chunks.map(chunk => ({
          word: chunk.text.trim(),
          startMs: Math.round(chunk.timestamp[0] * 1000),
          endMs: Math.round(chunk.timestamp[1] * 1000)
        })).filter(w => w.word.length > 0);

        const text = this.wordsWithTimestamps.map(w => w.word).join(' ');
        this.autoRecognizedText = text;
        if (this.customTranscriptInput) {
          this.customTranscriptInput.value = text;
        }
        
        this.updateMetricsUI();
        this.renderAnnotatedTranscript();
      } else {
        const text = output.text.trim();
        if (text.length > 0) {
          this.autoRecognizedText = text;
          if (this.customTranscriptInput) {
            this.customTranscriptInput.value = text;
          }
          this.onCustomTranscriptEdited();
        }
      }
      
    } catch (err) {
      console.error("Transcription error:", err);
      if (this.transcriptDisplay) {
        this.transcriptDisplay.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9em;">❌ Error transcribing audio. See console for details.</span>`;
      }
    } finally {
      this.isTranscribing = false;
    }
  }

  async toggleMicRecording() {
    if (!this.isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);
        this.audioChunks = [];

        this.mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) this.audioChunks.push(e.data);
        };

        this.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          const arrayBuffer = await audioBlob.arrayBuffer();
          const ctx = this.engine.getAudioContext();
          if (ctx.state === 'suspended') await ctx.resume();
          this.audioBuffer = await this.engine.decodeAudioData(arrayBuffer);
          this.currentFilename.textContent = "microphone_recording.wav";
          this.processAudioBuffer();
          if (!this.customTranscriptInput || !this.customTranscriptInput.value.trim()) {
            this.autoTranscribeAudio();
          }
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        this.btnRecordMic.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg> Stop Recording`;
        this.btnRecordMic.classList.add('btn-accent');
      } catch (err) {
        alert("Microphone access error: " + err.message);
      }
    } else {
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.btnRecordMic.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="6"/></svg> Record Mic`;
    }
  }

  processAudioBuffer() {
    if (!this.audioBuffer) return;

    // Run acoustic & pause analysis
    this.analysisResult = this.engine.analyzeAcoustics(this.audioBuffer);

    // If we already have exact word timestamps (e.g., from Demo), keep them
    if (this.wordsWithTimestamps && this.wordsWithTimestamps.length > 0) {
      // keep existing
    } else if (this.autoRecognizedText && this.autoRecognizedText.length > 0) {
      this.generateWordTimestampsFromText(this.autoRecognizedText, this.analysisResult.speechSegments);
    } else {
      this.generateWordTimestampsFromSpeechSegments(this.analysisResult.speechSegments);
    }

    this.updateMetricsUI();
    this.renderAnnotatedTranscript();
    this.renderPauseLocationsTable();
    this.renderWaveform();

    this.playerCard.style.display = 'block';
    this.currentDuration.textContent = `${this.formatTimeMs(this.analysisResult.durationMs)} (${this.analysisResult.durationMs} ms)`;
    this.stopPlayback();
  }

  onCustomTranscriptEdited() {
    if (!this.analysisResult) return;
    const text = this.customTranscriptInput ? this.customTranscriptInput.value.trim() : '';
    if (text.length > 0) {
      this.generateWordTimestampsFromText(text, this.analysisResult.speechSegments);
    } else {
      this.generateWordTimestampsFromSpeechSegments(this.analysisResult.speechSegments);
    }
    this.updateMetricsUI();
    this.renderAnnotatedTranscript();
  }

  generateWordTimestampsFromText(text, speechSegments) {
    const rawWords = text.split(/\s+/).filter(Boolean);
    if (rawWords.length === 0) {
      this.wordsWithTimestamps = [];
      return;
    }

    if (!speechSegments || speechSegments.length === 0) {
      const durationMs = this.analysisResult.durationMs;
      const step = durationMs / rawWords.length;
      this.wordsWithTimestamps = rawWords.map((word, i) => ({
        word,
        startMs: Math.round(i * step),
        endMs: Math.round((i + 1) * step - 20)
      }));
      return;
    }

    const totalSpeechDurationMs = speechSegments.reduce((acc, seg) => acc + seg.durationMs, 0);
    
    // Weight by length of word to give longer words more time
    const wordWeights = rawWords.map(w => w.length);
    const totalWeight = wordWeights.reduce((a, b) => a + b, 0);

    const wordsWithTimestamps = [];
    let currentSegIdx = 0;
    let currentSegUsedMs = 0;

    rawWords.forEach((word, idx) => {
      const wordSpeechTimeMs = (wordWeights[idx] / totalWeight) * totalSpeechDurationMs;
      let remainingWordTime = wordSpeechTimeMs;
      
      let startMs = -1;
      let endMs = -1;

      while (remainingWordTime > 0 && currentSegIdx < speechSegments.length) {
        const seg = speechSegments[currentSegIdx];
        const availableSegTime = seg.durationMs - currentSegUsedMs;

        if (startMs === -1) {
          startMs = seg.startMs + currentSegUsedMs;
        }

        if (availableSegTime >= remainingWordTime) {
          currentSegUsedMs += remainingWordTime;
          endMs = seg.startMs + currentSegUsedMs;
          remainingWordTime = 0;
        } else {
          remainingWordTime -= availableSegTime;
          currentSegIdx++;
          currentSegUsedMs = 0;
          endMs = seg.endMs;
        }
      }

      if (startMs === -1) {
        const lastSeg = speechSegments[speechSegments.length - 1];
        startMs = lastSeg.endMs - 10;
        endMs = lastSeg.endMs;
      }

      wordsWithTimestamps.push({
        word,
        startMs: Math.round(startMs),
        endMs: Math.max(Math.round(startMs) + 10, Math.round(endMs) - 15)
      });
    });

    this.wordsWithTimestamps = wordsWithTimestamps;
  }

  generateWordTimestampsFromSpeechSegments(speechSegments) {
    // Render clean prompt asking to type or auto-transcribe spoken audio
    const words = [];
    speechSegments.forEach((seg, idx) => {
      words.push({
        word: `(spoken word ${idx + 1})`,
        startMs: seg.startMs,
        endMs: seg.endMs
      });
    });
    this.wordsWithTimestamps = words;
  }

  updateMetricsUI() {
    // UI deleted in minimal mode
  }

  renderAnnotatedTranscript() {
    if (!this.analysisResult) return;
    const text = this.engine.buildAnnotatedTranscript(
      this.wordsWithTimestamps,
      this.analysisResult.pauseSegments,
      this.displayMinPauseMs
    );

    const tokens = text.split(/(\(pause detected: \d+ ms.*?\))/g);

    let html = '';
    let wordIdx = 0;

    tokens.forEach(tok => {
      if (!tok) return;
      if (tok.startsWith('(pause detected:')) {
        if (this.showPauses) {
          const match = tok.match(/\(pause detected: (\d+) ms(.*?)\)/);
          const duration = match ? match[1] : '';
          html += `<span class="pause-tag" style="font-size: 0.7em; color: var(--text-muted);">(pause: ${duration}ms)</span> `;
        }
      } else {
        const words = tok.trim().split(/\s+/).filter(Boolean);
        words.forEach(w => {
          html += `<span class="word-token" data-word-idx="${wordIdx}">${w}</span> `;
          wordIdx++;
        });
      }
    });

    if (this.transcriptDisplay) {
        this.transcriptDisplay.innerHTML = html;
    }
  }

  renderPauseLocationsTable() {
    // UI deleted in minimal mode
  }

  renderWaveform() {
    if (!this.analysisResult) return;

    const width = this.canvas.width;
    const height = this.canvas.height;
    this.ctx.clearRect(0, 0, width, height);

    const { frameEnergies, durationMs } = this.analysisResult;
    const numFrames = frameEnergies.length;
    const frameWidth = width / numFrames;

    for (let f = 0; f < numFrames; f++) {
      const x = f * frameWidth;
      const rms = frameEnergies[f];
      const barHeight = Math.max(4, rms * height * 2.5);
      const y = (height - barHeight) / 2;

      const isSpeech = rms >= this.engine.silenceThreshold;
      this.ctx.fillStyle = isSpeech ? '#10b981' : '#f59e0b';
      this.ctx.fillRect(x, y, Math.max(1, frameWidth - 0.5), barHeight);
    }

    const currentMs = this.getCurrentPlaybackMs();
    const playheadX = (currentMs / durationMs) * width;
    
    this.ctx.strokeStyle = '#00f2fe';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(playheadX, 0);
    this.ctx.lineTo(playheadX, height);
    this.ctx.stroke();
  }

  togglePlayPause() {
    if (this.isPlaying) {
      this.pausePlayback();
    } else {
      this.startPlayback(this.pausedAt);
    }
  }

  startPlayback(offsetSec = 0) {
    if (!this.audioBuffer) return;

    const ctx = this.engine.getAudioContext();
    if (this.audioSource) {
      this.audioSource.stop();
    }

    this.audioSource = ctx.createBufferSource();
    this.audioSource.buffer = this.audioBuffer;
    this.audioSource.connect(ctx.destination);

    this.startTime = ctx.currentTime - offsetSec;
    this.audioSource.start(0, offsetSec);
    this.isPlaying = true;

    this.iconPlay.style.display = 'none';
    this.iconPause.style.display = 'block';

    this.updatePlaybackLoop();
  }

  pausePlayback() {
    if (!this.isPlaying) return;
    const ctx = this.engine.getAudioContext();
    this.pausedAt = ctx.currentTime - this.startTime;
    if (this.audioSource) {
      this.audioSource.stop();
      this.audioSource = null;
    }
    this.isPlaying = false;
    this.iconPlay.style.display = 'block';
    this.iconPause.style.display = 'none';
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this.btnTranscribeSpeech) {
      this.btnTranscribeSpeech.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg> Auto-Transcribe Speech`;
    }
  }

  stopPlayback() {
    this.pausePlayback();
    this.pausedAt = 0;
    this.playTime.textContent = "00:00.000";
    this.renderWaveform();
  }

  seekToMs(ms) {
    const sec = ms / 1000;
    this.pausedAt = sec;
    if (this.isPlaying) {
      this.startPlayback(sec);
    } else {
      this.playTime.textContent = this.formatTimeMs(ms);
      this.renderWaveform();
      this.highlightActiveWord(ms);
    }
  }

  seekFromCanvas(e) {
    if (!this.audioBuffer) return;
    const rect = this.canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    const ms = ratio * (this.audioBuffer.duration * 1000);
    this.seekToMs(ms);
  }

  getCurrentPlaybackMs() {
    if (!this.isPlaying) return this.pausedAt * 1000;
    const ctx = this.engine.getAudioContext();
    const elapsedSec = ctx.currentTime - this.startTime;
    if (elapsedSec >= this.audioBuffer.duration) {
      this.stopPlayback();
      return 0;
    }
    return elapsedSec * 1000;
  }

  updatePlaybackLoop() {
    if (!this.isPlaying) return;
    const currentMs = this.getCurrentPlaybackMs();
    this.playTime.textContent = this.formatTimeMs(currentMs);
    this.renderWaveform();
    this.highlightActiveWord(currentMs);
    this.animFrameId = requestAnimationFrame(() => this.updatePlaybackLoop());
  }

  highlightActiveWord(currentMs) {
    const wordTokens = this.transcriptDisplay.querySelectorAll('.word-token');
    wordTokens.forEach((el) => {
      const idx = parseInt(el.getAttribute('data-word-idx'), 10);
      if (this.wordsWithTimestamps[idx]) {
        const { startMs, endMs } = this.wordsWithTimestamps[idx];
        if (currentMs >= startMs && currentMs <= endMs) {
          el.classList.add('active-word');
        } else {
          el.classList.remove('active-word');
        }
      }
    });
  }

  formatTimeMs(ms) {
    const totalSec = Math.floor(ms / 1000);
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    const millis = Math.floor(ms % 1000);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
  }

  copyTranscript() {
    const text = this.engine.buildAnnotatedTranscript(
      this.wordsWithTimestamps,
      this.analysisResult.pauseSegments,
      this.displayMinPauseMs
    );
    navigator.clipboard.writeText(text);
    alert("Transcript with inline pause annotations copied to clipboard!");
  }

  exportJsonReport() {
    const activePauses = this.analysisResult.pauseSegments.filter(p => p.durationMs >= this.displayMinPauseMs);
    const data = {
      filename: this.currentFilename.textContent,
      audioMetrics: {
        totalDurationMs: this.analysisResult.durationMs,
        totalSpeechTimeMs: this.analysisResult.totalSpeechMs,
        totalSilenceTimeMs: this.analysisResult.totalSilenceMs,
        syllableCount: this.engine.countTotalSyllables(this.wordsWithTimestamps.map(w => w.word).join(' ')),
        wordCount: this.wordsWithTimestamps.length,
        pauseCount: activePauses.length,
        displayMinPauseThresholdMs: this.displayMinPauseMs
      },
      annotatedTranscript: this.engine.buildAnnotatedTranscript(
        this.wordsWithTimestamps,
        this.analysisResult.pauseSegments,
        this.displayMinPauseMs
      ),
      locationOfPauses: activePauses.map((p, idx) => ({
        index: idx + 1,
        startMs: p.startMs,
        endMs: p.endMs,
        durationMs: p.durationMs,
        acousticClassification: p.eventType
      }))
    };

    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio_acoustic_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async runFullPipeline() {
    if (!this.audioBuffer) {
      alert("Please load or record some audio first!");
      return;
    }
    if (this.isPipelineRunning) return;
    this.isPipelineRunning = true;
    
    const originalBtnText = this.btnRunPipeline.innerHTML;
    
    try {
      this.btnRunPipeline.innerHTML = `<div class="spinner" style="width: 14px; height: 14px; display: inline-block; vertical-align: middle;"></div> Processing...`;
      this.pipelineSttOutput.textContent = "1. Enhancing audio...";
      this.pipelineLlmOutput.textContent = "...";
      
      let processingBuffer = this.audioBuffer;
      const isVoiceFocusOn = this.toggleVoiceFocus.checked;
      
      if (isVoiceFocusOn) {
        processingBuffer = await this.engine.enhanceAudioBuffer(this.audioBuffer);
      }
      
      this.pipelineSttOutput.textContent = "2. Transcribing with Whisper...";
      
      // STT
      const offlineCtx = new window.OfflineAudioContext(1, processingBuffer.duration * 16000, 16000);
      const source = offlineCtx.createBufferSource();
      source.buffer = processingBuffer;
      source.connect(offlineCtx.destination);
      source.start();
      const resampledBuffer = await offlineCtx.startRendering();
      const audioData = resampledBuffer.getChannelData(0);

      if (!this.transcriber) {
        this.pipelineSttOutput.textContent = "Downloading Whisper Model (once)...";
        this.transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
      }
      
      this.pipelineSttOutput.textContent = "Transcribing...";
      const sttOutput = await this.transcriber(audioData, { chunk_length_s: 30, stride_length_s: 5 });
      const transcript = sttOutput.text.trim();
      
      this.pipelineSttOutput.textContent = transcript || "(No speech detected)";
      
      if (!transcript) {
         this.pipelineLlmOutput.textContent = "No speech detected to answer.";
         return;
      }
      
      this.pipelineLlmOutput.textContent = "3. Generating response with LLM...";
      
      // LLM
      if (!this.llmGenerator) {
        this.pipelineLlmOutput.textContent = "Downloading LaMini-Flan-T5-77M Model (77MB, once)...";
        this.llmGenerator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-77M');
      }
      
      this.pipelineLlmOutput.textContent = "Thinking...";
      const prompt = `Summarize the following text in one short sentence:\n\n${transcript}\n\nSummary:`;
      
      const llmOutput = await this.llmGenerator(prompt, { max_new_tokens: 50 });
      const responseText = llmOutput[0].generated_text.trim();
      
      this.pipelineLlmOutput.textContent = responseText;
      
      // TTS
      this.pipelineLlmOutput.innerHTML += `<br><br><span style="color:var(--accent-purple);font-size:0.85em;">🗣️ 4. Speaking...</span>`;
      const utterance = new SpeechSynthesisUtterance(responseText);
      window.speechSynthesis.speak(utterance);
      
    } catch (e) {
      console.error(e);
      this.pipelineSttOutput.textContent = "Pipeline Error: " + e.message;
    } finally {
      this.isPipelineRunning = false;
      this.btnRunPipeline.innerHTML = originalBtnText;
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
