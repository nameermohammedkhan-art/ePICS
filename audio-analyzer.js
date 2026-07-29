/**
 * Audio Analyzer Engine V1 (Refined)
 * Speech-to-Text, Pause Detection (ms), Acoustic Event Classification & Syllable/Word Metrics
 */

export class AudioAnalyzerEngine {
  constructor(options = {}) {
    this.audioCtx = null;
    this.minPauseDurationMs = options.minPauseDurationMs || 50; // Min duration to qualify as acoustic pause
    this.displayMinPauseMs = options.displayMinPauseMs || 100; // Min duration to display inline in transcript
    this.frameSizeMs = options.frameSizeMs || 5; // 5ms analysis frames
  }

  getAudioContext() {
    if (!this.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  /**
   * Decodes an ArrayBuffer or Blob into AudioBuffer
   */
  async decodeAudioData(arrayBuffer) {
    const ctx = this.getAudioContext();
    const bufferCopy = arrayBuffer.slice(0);
    return await ctx.decodeAudioData(bufferCopy);
  }

  /**
   * Main acoustic analysis method for an AudioBuffer
   */
  analyzeAcoustics(audioBuffer, customOptions = {}) {
    const minPauseDurationMs = customOptions.minPauseDurationMs !== undefined ? customOptions.minPauseDurationMs : this.minPauseDurationMs;

    const sampleRate = audioBuffer.sampleRate;
    const numberOfChannels = audioBuffer.numberOfChannels;
    const durationSec = audioBuffer.duration;
    const durationMs = Math.round(durationSec * 1000);

    // Merge multi-channel down to mono float array
    const pcm = new Float32Array(audioBuffer.length);
    for (let c = 0; c < numberOfChannels; c++) {
      const channelData = audioBuffer.getChannelData(c);
      for (let i = 0; i < audioBuffer.length; i++) {
        pcm[i] += channelData[i] / numberOfChannels;
      }
    }

    // Frame-based energy analysis
    const samplesPerFrame = Math.floor((this.frameSizeMs / 1000) * sampleRate);
    const totalFrames = Math.floor(pcm.length / samplesPerFrame);

    const frameEnergies = new Float32Array(totalFrames);
    const frameStatus = new Array(totalFrames); // 'speech' | 'silence'

    // Compute RMS Energy for each frame
    let maxEnergy = 0;
    for (let f = 0; f < totalFrames; f++) {
      let sumSquare = 0;
      const start = f * samplesPerFrame;
      for (let i = 0; i < samplesPerFrame; i++) {
        const val = pcm[start + i];
        sumSquare += val * val;
      }
      const rms = Math.sqrt(sumSquare / samplesPerFrame);
      frameEnergies[f] = rms;
      if (rms > maxEnergy) maxEnergy = rms;
    }

    // Dynamic threshold adapting to signal energy
    const sortedEnergies = new Float32Array(frameEnergies).sort();
    const p10 = sortedEnergies[Math.floor(sortedEnergies.length * 0.1)]; // Noise floor estimate
    const p50 = sortedEnergies[Math.floor(sortedEnergies.length * 0.5)]; // Median energy
    // Threshold sits comfortably above the noise floor
    const adaptiveThreshold = p10 + (p50 - p10) * 0.5;

    // Frame classification
    for (let f = 0; f < totalFrames; f++) {
      frameStatus[f] = frameEnergies[f] >= adaptiveThreshold ? 'speech' : 'silence';
    }

    // Temporal Hysteresis & Smoothing: fill tiny gaps < 60ms in speech
    const gapFillFrames = Math.floor(60 / this.frameSizeMs);
    for (let f = 1; f < totalFrames - gapFillFrames; f++) {
      if (frameStatus[f] === 'silence') {
        let isNextSpeech = false;
        for (let k = 1; k <= gapFillFrames; k++) {
          if (frameStatus[f + k] === 'speech') {
            isNextSpeech = true;
            break;
          }
        }
        if (frameStatus[f - 1] === 'speech' && isNextSpeech) {
          frameStatus[f] = 'speech';
        }
      }
    }

    // Segment calculation
    let totalSpeechMs = 0;
    let totalSilenceMs = 0;
    const pauseSegments = [];
    const speechSegments = [];

    let currentSegment = null;

    for (let f = 0; f < totalFrames; f++) {
      const timeMs = Math.round((f * samplesPerFrame / sampleRate) * 1000);
      const status = frameStatus[f];
      const frameDuration = Math.round((samplesPerFrame / sampleRate) * 1000);

      if (!currentSegment || currentSegment.type !== status) {
        if (currentSegment) {
          currentSegment.endMs = timeMs;
          currentSegment.durationMs = currentSegment.endMs - currentSegment.startMs;
          
          if (currentSegment.type === 'speech') {
            totalSpeechMs += currentSegment.durationMs;
            speechSegments.push(currentSegment);
          } else {
            totalSilenceMs += currentSegment.durationMs;
            if (currentSegment.durationMs >= minPauseDurationMs) {
              const bgEvent = this.classifyBackgroundAcoustics(
                pcm,
                Math.floor((currentSegment.startMs / 1000) * sampleRate),
                Math.floor((currentSegment.endMs / 1000) * sampleRate),
                sampleRate
              );
              currentSegment.eventType = bgEvent;
              pauseSegments.push(currentSegment);
            }
          }
        }

        currentSegment = {
          type: status,
          startMs: timeMs,
          endMs: timeMs + frameDuration,
          durationMs: frameDuration
        };
      }
    }

    if (currentSegment) {
      currentSegment.endMs = durationMs;
      currentSegment.durationMs = currentSegment.endMs - currentSegment.startMs;
      if (currentSegment.type === 'speech') {
        totalSpeechMs += currentSegment.durationMs;
        speechSegments.push(currentSegment);
      } else {
        totalSilenceMs += currentSegment.durationMs;
        if (currentSegment.durationMs >= minPauseDurationMs) {
          const bgEvent = this.classifyBackgroundAcoustics(
            pcm,
            Math.floor((currentSegment.startMs / 1000) * sampleRate),
            Math.floor((currentSegment.endMs / 1000) * sampleRate),
            sampleRate
          );
          currentSegment.eventType = bgEvent;
          pauseSegments.push(currentSegment);
        }
      }
    }

    // Exact silence duration calculation
    totalSilenceMs = durationMs - totalSpeechMs;

    return {
      durationMs,
      totalSpeechMs,
      totalSilenceMs,
      pauseSegments,
      speechSegments,
      frameEnergies,
      pcm,
      sampleRate
    };
  }

  /**
   * Classifies background sound events in non-speech intervals
   */
  classifyBackgroundAcoustics(pcm, startSample, endSample, sampleRate) {
    const length = Math.max(0, endSample - startSample);
    if (length < 64) return 'Clean Silence';

    let sum = 0;
    let maxAmp = 0;
    let zeroCrossings = 0;

    for (let i = startSample; i < endSample && i < pcm.length; i++) {
      const val = pcm[i];
      const absVal = Math.abs(val);
      sum += absVal;
      if (absVal > maxAmp) maxAmp = absVal;
      if (i > startSample && ((pcm[i] >= 0 && pcm[i - 1] < 0) || (pcm[i] < 0 && pcm[i - 1] >= 0))) {
        zeroCrossings++;
      }
    }

    const avgAmp = sum / length;
    const zcr = zeroCrossings / (length / sampleRate);

    // Only tag non-speech sound when amplitude is significantly above baseline silence
    if (maxAmp < 0.015) {
      return 'Clean Silence';
    } else if (zcr > 3500 && avgAmp > 0.02) {
      return 'Background Noise / Hiss';
    } else if (maxAmp > 0.1 && avgAmp < 0.025) {
      return 'Click / Snap';
    } else if (avgAmp > 0.025) {
      return 'Background Hum / Audio';
    }
    return 'Clean Silence';
  }

  /**
   * Syllable Counter Engine
   */
  countSyllablesInWord(word) {
    if (!word) return 0;
    let w = word.toLowerCase().replace(/[^a-z]/g, '');
    if (w.length === 0) return 0;
    if (w.length <= 3) return 1;

    w = w.replace(/(?:[^laeiouy]es|ed|e)$/, '');
    w = w.replace(/^y/, '');
    const syllables = w.match(/[aeiouy]{1,2}/g);
    return syllables ? syllables.length : 1;
  }

  countTotalSyllables(text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    return words.reduce((acc, word) => acc + this.countSyllablesInWord(word), 0);
  }

  countWords(text) {
    const words = text.trim().split(/\s+/).filter(w => w.replace(/[^a-zA-Z0-9']/g, '').length > 0);
    return words.length;
  }

  /**
   * Formats annotated transcript inserting exact pause tags into text
   * Filter out micro-pauses below displayMinPauseMs so the transcript is clean and readable
   */
  buildAnnotatedTranscript(wordsWithTimestamps, pauseSegments, displayMinPauseMs = 100, speechSegments = []) {
    if (!wordsWithTimestamps || wordsWithTimestamps.length === 0) {
      return '';
    }

    const visiblePauses = pauseSegments.filter(p => p.durationMs >= displayMinPauseMs);

    // If speech segments are available, use the Production-Grade Speech Chunk Merge Strategy
    if (speechSegments && speechSegments.length > 0) {
      // 1. Initialize speech chunks with support for rich word objects
      const chunks = speechSegments.map(seg => ({
        ...seg,
        words: []
      }));

      // 2. Assign every word to exactly one speech chunk using maximum overlap.
      // With WhisperX forced alignment, word boundaries are precise — overlap is reliable.
      wordsWithTimestamps.forEach(w => {
        let bestChunkIdx = -1;
        let maxOverlap = 0;

        // Primary: assign by largest physical overlap with a speech chunk
        chunks.forEach((chunk, idx) => {
          const overlapStart = Math.max(w.startMs, chunk.startMs);
          const overlapEnd = Math.min(w.endMs, chunk.endMs);
          const overlap = overlapEnd - overlapStart; // negative = no overlap
          if (overlap > maxOverlap) {
            maxOverlap = overlap;
            bestChunkIdx = idx;
          }
        });

        // Fallback: word lies in a gap (rare with forced alignment).
        // Find closest chunk by boundary distance. Log if drift > 80ms.
        if (bestChunkIdx === -1) {
          const midpoint = (w.startMs + w.endMs) / 2;
          let minDist = Infinity;
          chunks.forEach((chunk, idx) => {
            const dist = Math.min(
              Math.abs(midpoint - chunk.startMs),
              Math.abs(midpoint - chunk.endMs)
            );
            if (dist < minDist) {
              minDist = dist;
              bestChunkIdx = idx;
            }
          });
          if (minDist > 80) {
            console.warn(`[ALIGN DRIFT] Word "${w.word}" (${w.startMs}–${w.endMs}ms) is ${Math.round(minDist)}ms from nearest VAD boundary. Assigned to chunk ${bestChunkIdx + 1}.`);
          }
        }

        chunks[bestChunkIdx].words.push(w);
      });

      // 3. Render strictly Chunk -> Pause -> Chunk
      const annotatedParts = [];
      chunks.forEach((chunk, idx) => {
        if (chunk.words.length > 0) {
          // Sort words chronologically inside the chunk to handle Whisper out-of-order timings
          chunk.words.sort((a, b) => a.startMs - b.startMs);
          const chunkText = chunk.words.map(w => w.word).join(' ');
          annotatedParts.push(chunkText);
        }

        // Find the pause that belongs between this chunk and the next chunk (gap-based matching)
        if (idx < chunks.length - 1) {
          const gapStart = chunk.endMs;
          const gapEnd = chunks[idx + 1].startMs;

          const pause = visiblePauses.find(p => 
            p.startMs <= gapEnd && p.endMs >= gapStart
          );

          if (pause) {
            const bgText = pause.eventType && pause.eventType !== 'Clean Silence' ? ` [${pause.eventType}]` : '';
            annotatedParts.push(`(pause detected: ${pause.durationMs} ms${bgText} | start: ${pause.startMs} | end: ${pause.endMs})`);
          }
        }
      });

      return annotatedParts.join(' ');
    }

    // Fallback sorting method for demo audio (when no backend VAD chunks exist)
    const events = [];

    wordsWithTimestamps.forEach((w) => {
      events.push({
        type: 'word',
        midpoint: (w.startMs + w.endMs) / 2,
        text: w.word
      });
    });

    visiblePauses.forEach(p => {
      events.push({
        type: 'pause',
        midpoint: (p.startMs + p.endMs) / 2,
        durationMs: p.durationMs,
        eventType: p.eventType,
        startMs: p.startMs,
        endMs: p.endMs
      });
    });

    events.sort((a, b) => {
      if (a.midpoint !== b.midpoint) {
        return a.midpoint - b.midpoint;
      }
      return a.type === 'pause' ? -1 : 1;
    });

    let annotatedParts = [];
    events.forEach(e => {
      if (e.type === 'word') {
        annotatedParts.push(e.text);
      } else {
        const bgText = e.eventType && e.eventType !== 'Clean Silence' ? ` [${e.eventType}]` : '';
        annotatedParts.push(`(pause detected: ${e.durationMs} ms${bgText} | start: ${e.startMs} | end: ${e.endMs})`);
      }
    });

    return annotatedParts.join(' ');
  }

  /**
   * Applies "Voice Focus" (audio enhancement) using Web Audio API filters.
   * This removes low rumble, high hiss, and compresses the dynamic range to isolate speech.
   */
  async enhanceAudioBuffer(audioBuffer) {
    const OfflineAudioCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const offlineCtx = new OfflineAudioCtx(
      audioBuffer.numberOfChannels,
      audioBuffer.duration * audioBuffer.sampleRate,
      audioBuffer.sampleRate
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;

    // 1. Highpass filter to remove low rumble (e.g., wind, mic handling)
    const highpass = offlineCtx.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = 200; // Human voice fundamental frequency

    // 2. Lowpass filter to remove high-frequency hiss
    const lowpass = offlineCtx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 3500; // Upper human voice range

    // 3. Dynamics Compressor to level out the volume
    const compressor = offlineCtx.createDynamicsCompressor();
    compressor.threshold.value = -35;
    compressor.knee.value = 40;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.25;

    // Gain node to make up for lost volume
    const makeupGain = offlineCtx.createGain();
    makeupGain.gain.value = 2.5;

    source.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(compressor);
    compressor.connect(makeupGain);
    makeupGain.connect(offlineCtx.destination);

    source.start(0);
    return await offlineCtx.startRendering();
  }
}
