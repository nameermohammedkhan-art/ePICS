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
    this.sampleBtns = document.querySelectorAll('.sample-btn');
    this.btnRecordMic = document.getElementById('btn-record-mic');

    this.canvas = document.getElementById('waveform-canvas');
    this.ctx = this.canvas.getContext('2d');

    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.iconPlay = document.getElementById('icon-play');
    this.iconPause = document.getElementById('icon-pause');
    this.playTime = document.getElementById('play-time');
    this.currentFilename = document.getElementById('current-filename');
    this.currentDuration = document.getElementById('current-duration');
    this.seekSlider = document.getElementById('seek-slider');

    this.transcriptDisplay = document.getElementById('transcript-box');
    this.showPauses = false;
    this.pauseCountStat = document.getElementById('pause-count-stat');
    this.displayMinPauseMs = 10; // Restore missing parameter
    
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
    if (this.sampleBtns) {
      this.sampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
          const type = btn.getAttribute('data-type');
          this.sampleBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.loadDemoAudio(type);
        });
      });
    }

    this.btnRecordMic.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMicRecording();
    });

    this.btnPlayPause.addEventListener('click', () => this.togglePlayPause());
    this.canvas.addEventListener('click', (e) => this.seekFromCanvas(e));

    if (this.seekSlider) {
      this.seekSlider.addEventListener('input', (e) => {
        if (!this.audioBuffer) return;
        const ratio = parseFloat(e.target.value) / 100;
        const ms = ratio * (this.audioBuffer.duration * 1000);
        this.seekToMs(ms);
      });
    }

    const vfOffBtn = document.getElementById('vf-off');
    const vfOnBtn = document.getElementById('vf-on');

    if (vfOffBtn && vfOnBtn) {
      vfOnBtn.addEventListener('click', () => {
        this.setPauseDetection(true);
      });

      vfOffBtn.addEventListener('click', () => {
        this.setPauseDetection(false);
      });
    }
  }

  setPauseDetection(visible) {
    this.showPauses = visible;
    
    const vfOffBtn = document.getElementById('vf-off');
    const vfOnBtn = document.getElementById('vf-on');
    
    if (vfOffBtn && vfOnBtn) {
      if (visible) {
        vfOnBtn.style.background = 'var(--text-main)';
        vfOnBtn.style.color = 'var(--bg-color)';
        vfOffBtn.style.background = 'var(--bg-color)';
        vfOffBtn.style.color = 'var(--text-main)';
      } else {
        vfOffBtn.style.background = 'var(--text-main)';
        vfOffBtn.style.color = 'var(--bg-color)';
        vfOnBtn.style.background = 'var(--bg-color)';
        vfOnBtn.style.color = 'var(--text-main)';
      }
    }
    
    if (this.analysisResult) {
      this.renderAnnotatedTranscript();
      this.updateMetricsUI();
    }
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = 50;
    this.renderWaveform();
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

  async loadDemoAudio(type = 'robot') {
    const ctx = this.engine.getAudioContext();
    const sampleRate = ctx.sampleRate;

    let totalDurationSec = 3.5;
    let filename = "robot_sample.wav";
    let speechSegments = [];
    let words = [];

    const msToSample = (ms) => Math.floor((ms / 1000) * sampleRate);

    if (type === 'robot') {
      totalDurationSec = 3.5;
      filename = "robot_sample.wav";
      speechSegments = [
        { start: 50, end: 300, freq: 180 },   // My
        { start: 340, end: 600, freq: 210 },  // name
        { start: 630, end: 800, freq: 195 },  // is
        { start: 840, end: 1200, freq: 175 }, // Raju
        { start: 1217, end: 1400, freq: 200 }, // I
        { start: 1440, end: 1650, freq: 190 }, // am
        { start: 1680, end: 1780, freq: 185 }, // a
        { start: 1820, end: 2400, freq: 170 }, // football
        { start: 2413, end: 3100, freq: 180 }  // player
      ];
      words = [
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
    } else if (type === 'portfolio') {
      totalDurationSec = 4.0;
      filename = "portfolio_sample.wav";
      speechSegments = [
        { start: 50, end: 250, freq: 220 },   // This
        { start: 280, end: 400, freq: 200 },  // is
        { start: 420, end: 600, freq: 210 },  // my
        { start: 620, end: 1000, freq: 180 }, // audio
        { start: 1020, end: 1600, freq: 175 }, // portfolio
        { start: 1650, end: 1800, freq: 200 }, // I
        { start: 1820, end: 2100, freq: 190 }, // hope
        { start: 2125, end: 2300, freq: 185 }, // you
        { start: 2325, end: 2700, freq: 170 }  // like it
      ];
      words = [
        { word: "This", startMs: 50, endMs: 250 },
        { word: "is", startMs: 280, endMs: 400 },
        { word: "my", startMs: 420, endMs: 600 },
        { word: "audio", startMs: 620, endMs: 1000 },
        { word: "portfolio.", startMs: 1020, endMs: 1600 },
        { word: "I", startMs: 1650, endMs: 1800 },
        { word: "hope", startMs: 1820, endMs: 2100 },
        { word: "you", startMs: 2125, endMs: 2300 },
        { word: "like it.", startMs: 2325, endMs: 2700 }
      ];
    } else if (type === 'incredible') {
      totalDurationSec = 4.0;
      filename = "incredible_sample.wav";
      speechSegments = [
        { start: 50, end: 350, freq: 190 },   // That
        { start: 380, end: 500, freq: 200 },  // is
        { start: 520, end: 900, freq: 210 },  // simply
        { start: 980, end: 1600, freq: 170 }, // incredible
        { start: 1680, end: 2000, freq: 180 }, // Look
        { start: 2020, end: 2200, freq: 190 }, // at
        { start: 2210, end: 2500, freq: 185 }, // those
        { start: 2515, end: 3200, freq: 160 }  // machines
      ];
      words = [
        { word: "That", startMs: 50, endMs: 350 },
        { word: "is", startMs: 380, endMs: 500 },
        { word: "simply", startMs: 520, endMs: 900 },
        { word: "incredible.", startMs: 980, endMs: 1600 },
        { word: "Look", startMs: 1680, endMs: 2000 },
        { word: "at", startMs: 2020, endMs: 2200 },
        { word: "those", startMs: 2210, endMs: 2500 },
        { word: "machines.", startMs: 2515, endMs: 3200 }
      ];
    } else if (type === 'machines') {
      totalDurationSec = 4.5;
      filename = "machines_sample.wav";
      speechSegments = [
        { start: 50, end: 300, freq: 170 },   // The
        { start: 330, end: 800, freq: 180 },  // machines
        { start: 830, end: 1000, freq: 190 }, // are
        { start: 1020, end: 1500, freq: 200 }, // running
        { start: 1620, end: 1850, freq: 195 }, // We
        { start: 1870, end: 2150, freq: 190 }, // must
        { start: 2165, end: 2600, freq: 185 }, // monitor
        { start: 2615, end: 3000, freq: 175 }  // them
      ];
      words = [
        { word: "The", startMs: 50, endMs: 300 },
        { word: "machines", startMs: 330, endMs: 800 },
        { word: "are", startMs: 830, endMs: 1000 },
        { word: "running.", startMs: 1020, endMs: 1500 },
        { word: "We", startMs: 1620, endMs: 1850 },
        { word: "must", startMs: 1870, endMs: 2150 },
        { word: "monitor", startMs: 2165, endMs: 2600 },
        { word: "them.", startMs: 2615, endMs: 3000 }
      ];
    }

    const length = Math.floor(sampleRate * totalDurationSec);
    const audioBuffer = ctx.createBuffer(1, length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);

    // Generate speech tones
    speechSegments.forEach(seg => {
      const startS = msToSample(seg.start);
      const endS = msToSample(seg.end);
      for (let i = startS; i < endS && i < length; i++) {
        const t = (i - startS) / sampleRate;
        const envelope = Math.sin(Math.PI * (i - startS) / (endS - startS));
        const wave = 0.3 * Math.sin(2 * Math.PI * seg.freq * t) +
                     0.15 * Math.sin(2 * Math.PI * (seg.freq * 2) * t) +
                     0.08 * Math.sin(2 * Math.PI * (seg.freq * 3) * t);
        channelData[i] = wave * envelope;
      }
    });

    this.wordsWithTimestamps = words;
    this.audioBuffer = audioBuffer;
    
    if (this.currentFilename) {
      this.currentFilename.textContent = filename;
    }
    
    // Set slider to 10ms for demo so 17ms and 13ms pauses display as specified
    this.displayMinPauseMs = 10;

    this.processAudioBuffer();
  }

  async autoTranscribeAudio() {
    if (!this.audioBuffer) return;

    if (this.isTranscribing) return;
    this.isTranscribing = true;

    if (this.transcriptDisplay) {
      this.transcriptDisplay.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; color: var(--text-muted); font-size: 0.95em;">
          <div class="spinner" style="width: 18px; height: 18px; border: 2px solid #ccc; border-top-color: var(--text-main); display: inline-block; flex-shrink: 0;"></div>
          <span>Connecting to local Whisper backend...</span>
        </div>
      `;
    }

    try {
      // Convert AudioBuffer to 16-bit PCM WAV Blob
      const offlineCtx = new window.OfflineAudioContext(1, this.audioBuffer.duration * 16000, 16000);
      const source = offlineCtx.createBufferSource();
      source.buffer = this.audioBuffer;
      source.connect(offlineCtx.destination);
      source.start();
      const resampledBuffer = await offlineCtx.startRendering();
      const audioData = resampledBuffer.getChannelData(0);
      
      const buffer = new ArrayBuffer(44 + audioData.length * 2);
      const view = new DataView(buffer);
      
      const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };
      
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + audioData.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, 16000, true);
      view.setUint32(28, 16000 * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, 'data');
      view.setUint32(40, audioData.length * 2, true);
      
      let offset = 44;
      for (let i = 0; i < audioData.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      
      const wavBlob = new Blob([buffer], { type: 'audio/wav' });

      if (this.transcriptDisplay) {
        this.transcriptDisplay.innerHTML = `
          <div style="display: flex; align-items: center; gap: 12px; color: var(--text-muted); font-size: 0.95em;">
            <div class="spinner" style="width: 18px; height: 18px; border: 2px solid #ccc; border-top-color: var(--text-main); display: inline-block; flex-shrink: 0;"></div>
            <span>Transcribing on local Whisper backend...</span>
          </div>
        `;
      }

      const formData = new FormData();
      formData.append('file', wavBlob, 'audio.wav');

      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      if (data.words) {
        this.wordsWithTimestamps = data.words;
        
        // Use VAD data from backend
        this.analysisResult = {
          durationMs: this.audioBuffer.duration * 1000,
          speechSegments: data.speech_segments || [],
          pauseSegments: data.pauses || []
        };
        
        const text = data.text;
        this.autoRecognizedText = text;
        if (this.customTranscriptInput) {
          this.customTranscriptInput.value = text;
        }
        
        this.processAudioBuffer();
        this.setPauseDetection(true);
      } else {
        throw new Error("Invalid response format from backend.");
      }
      
    } catch (err) {
      console.error("Transcription error:", err);
      if (this.transcriptDisplay) {
        this.transcriptDisplay.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9em;">❌ Error transcribing audio. Is the Python backend running on port 8000?</span>`;
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

    // Run acoustic analysis ONLY to get frameEnergies for the visual waveform rendering.
    // We completely ignore its pause/speech detection.
    const acousticResult = this.engine.analyzeAcoustics(this.audioBuffer);

    if (!this.analysisResult) {
      this.analysisResult = {
        durationMs: this.audioBuffer.duration * 1000,
        speechSegments: [],
        pauseSegments: [],
        frameEnergies: acousticResult.frameEnergies
      };
    } else {
      this.analysisResult.frameEnergies = acousticResult.frameEnergies;
    }

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

    if (this.playerCard) {
      this.playerCard.style.display = 'block';
    }
    if (this.currentDuration) {
      this.currentDuration.textContent = this.formatTimeShort(this.analysisResult.durationMs);
    }
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
    if (!this.analysisResult) return;
    const activePauses = this.analysisResult.pauseSegments.filter(p => p.durationMs >= this.displayMinPauseMs);
    const pauseCount = this.showPauses ? activePauses.length : 0;
    if (this.pauseCountStat) {
      this.pauseCountStat.textContent = pauseCount;
    }
  }

  renderAnnotatedTranscript() {
    if (!this.analysisResult) return;
    const text = this.engine.buildAnnotatedTranscript(
      this.wordsWithTimestamps,
      this.analysisResult.pauseSegments,
      this.displayMinPauseMs,
      this.analysisResult.speechSegments
    );

    const tokens = text.split(/(\(pause detected: \d+ ms.*?\))/g);
    let html = '';
    let wordIdx = 0;

    tokens.forEach(tok => {
      if (!tok) return;
      if (tok.startsWith('(pause detected:')) {
        if (this.showPauses) {
          const match = tok.match(/\(pause detected: (\d+) ms.*?\| start: (\d+) \| end: (\d+)\)/);
          const duration = match ? match[1] : '';
          const startMs = match ? parseInt(match[2]) : 0;
          const endMs = match ? parseInt(match[3]) : 0;
          html += `<span class="pause-tag" data-start-ms="${startMs}" data-end-ms="${endMs}" style="font-size: 0.7em; color: var(--text-muted); transition: background 0.15s ease;">(pause: ${duration}ms)</span> `;
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
    const width = this.canvas.width;
    const height = this.canvas.height;
    this.ctx.clearRect(0, 0, width, height);

    if (!this.analysisResult) {
      // Draw flat baseline
      this.ctx.strokeStyle = 'var(--pause-color)';
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      this.ctx.moveTo(0, height / 2);
      this.ctx.lineTo(width, height / 2);
      this.ctx.stroke();
      return;
    }

    const { frameEnergies, durationMs } = this.analysisResult;
    const numFrames = frameEnergies.length;
    const frameWidth = width / numFrames;

    for (let f = 0; f < numFrames; f++) {
      const x = f * frameWidth;
      const rms = frameEnergies[f];
      const barHeight = Math.max(2, rms * height * 2.5);
      const y = (height - barHeight) / 2;

      const isSpeech = rms >= this.silenceThreshold;
      this.ctx.fillStyle = isSpeech ? 'var(--text-main)' : 'var(--pause-color)';
      this.ctx.fillRect(x, y, Math.max(1, frameWidth - 0.5), barHeight);
    }

    const currentMs = this.getCurrentPlaybackMs();
    const playheadX = (currentMs / durationMs) * width;
    
    this.ctx.strokeStyle = 'var(--text-main)';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(playheadX, 0);
    this.ctx.lineTo(playheadX, height);
    this.ctx.stroke();
  }

  togglePlayPause() {
    try {
      if (this.isPlaying) {
        this.pausePlayback();
      } else {
        this.startPlayback(this.pausedAt);
      }
    } catch (e) {
      alert("Play/Pause error: " + e.message);
    }
  }

  async startPlayback(offsetSec = 0) {
    try {
      if (!this.audioBuffer) return;

      const ctx = this.engine.getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      if (this.audioSource) {
        try {
          this.audioSource.stop();
        } catch (e) {}
      }

      this.audioSource = ctx.createBufferSource();
      this.audioSource.buffer = this.audioBuffer;
      this.audioSource.connect(ctx.destination);

      this.startTime = ctx.currentTime - offsetSec;
      this.audioSource.start(0, offsetSec);
      this.isPlaying = true;

      if (this.iconPlay) this.iconPlay.style.display = 'none';
      if (this.iconPause) this.iconPause.style.display = 'block';

      this.updatePlaybackLoop();
    } catch (e) {
      alert("Start playback error: " + e.message);
    }
  }

  pausePlayback() {
    if (!this.isPlaying) return;
    const ctx = this.engine.getAudioContext();
    this.pausedAt = ctx.currentTime - this.startTime;
    if (this.audioSource) {
      try {
        this.audioSource.stop();
      } catch (e) {}
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
    if (this.playTime) {
      this.playTime.textContent = "0:00";
    }
    if (this.seekSlider) {
      this.seekSlider.value = 0;
    }
    this.renderWaveform();
  }

  seekToMs(ms) {
    const sec = ms / 1000;
    this.pausedAt = sec;
    
    if (this.seekSlider && this.audioBuffer) {
      const duration = this.audioBuffer.duration;
      this.seekSlider.value = (ms / (duration * 1000)) * 100;
    }

    if (this.isPlaying) {
      this.startPlayback(sec);
    } else {
      if (this.playTime) {
        this.playTime.textContent = this.formatTimeShort(ms);
      }
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
    if (this.playTime) {
      this.playTime.textContent = this.formatTimeShort(currentMs);
    }
    this.renderWaveform();
    this.highlightActiveWord(currentMs);
    
    if (this.seekSlider && this.audioBuffer) {
      const duration = this.audioBuffer.duration;
      this.seekSlider.value = (currentMs / (duration * 1000)) * 100;
    }

    this.animFrameId = requestAnimationFrame(() => this.updatePlaybackLoop());
  }

  highlightActiveWord(currentMs) {
    let isPauseActive = false;

    const pauseTokens = this.transcriptDisplay.querySelectorAll('.pause-tag');
    pauseTokens.forEach((el) => {
      const startMs = parseFloat(el.getAttribute('data-start-ms'));
      const endMs = parseFloat(el.getAttribute('data-end-ms'));
      if (currentMs >= startMs && currentMs <= endMs) {
        el.classList.add('active-word');
        isPauseActive = true;
      } else {
        el.classList.remove('active-word');
      }
    });

    const wordTokens = this.transcriptDisplay.querySelectorAll('.word-token');
    wordTokens.forEach((el) => {
      const idx = parseInt(el.getAttribute('data-word-idx'), 10);
      if (this.wordsWithTimestamps[idx]) {
        const { startMs, endMs } = this.wordsWithTimestamps[idx];
        if (!isPauseActive && currentMs >= startMs && currentMs <= endMs) {
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

  formatTimeShort(ms) {
    const totalSec = Math.floor(ms / 1000);
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
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
