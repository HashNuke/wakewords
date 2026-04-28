const MODEL_SAMPLE_RATE = 16000;
const N_FFT = 512;
const WIN_LENGTH = 400;
const HOP_LENGTH = 160;
const N_MELS = 64;
const N_MFCC = 64;
const LOG_MEL_EPSILON = 1e-6;

let mfccCache = null;

export function mfccFeatures(samples) {
  const cache = getMfccCache();
  const padded = reflectPad(samples, N_FFT / 2);
  const frames = Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1;
  const mfcc = new Float32Array(N_MFCC * frames);
  const spectrum = new Array(N_FFT / 2 + 1).fill(0);
  const mel = new Array(N_MELS).fill(0);
  const frameSamples = new Array(N_FFT).fill(0);

  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * HOP_LENGTH;
    for (let index = 0; index < N_FFT; index += 1) {
      frameSamples[index] = padded[offset + index] * cache.window[index];
    }
    for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
      let real = 0;
      let imag = 0;
      const cosRow = cache.cos[bin];
      const sinRow = cache.sin[bin];
      for (let index = 0; index < N_FFT; index += 1) {
        const value = frameSamples[index];
        real += value * cosRow[index];
        imag -= value * sinRow[index];
      }
      spectrum[bin] = real * real + imag * imag;
    }

    for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
      let energy = 0;
      const filter = cache.melFilters[melIndex];
      for (const [bin, weight] of filter) {
        energy += spectrum[bin] * weight;
      }
      mel[melIndex] = Math.log(energy + LOG_MEL_EPSILON);
    }

    for (let coeff = 0; coeff < N_MFCC; coeff += 1) {
      let value = 0;
      for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
        value += mel[melIndex] * cache.dct[melIndex][coeff];
      }
      mfcc[coeff * frames + frame] = value;
    }
  }

  return { data: mfcc, features: N_MFCC, frames };
}

export function resample(samples, sourceRate, targetRate = MODEL_SAMPLE_RATE) {
  if (sourceRate === targetRate) return new Float32Array(samples);
  const targetLength = Math.max(1, Math.round((samples.length * targetRate) / sourceRate));
  const result = new Float32Array(targetLength);
  const scale = (samples.length - 1) / Math.max(1, targetLength - 1);
  for (let index = 0; index < targetLength; index += 1) {
    const position = index * scale;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const weight = position - left;
    result[index] = samples[left] * (1 - weight) + samples[right] * weight;
  }
  return result;
}

function getMfccCache() {
  if (mfccCache) return mfccCache;
  mfccCache = {
    window: paddedHannWindow(),
    cos: dftTable(Math.cos),
    sin: dftTable(Math.sin),
    melFilters: melFilterbank(),
    dct: dctMatrix(),
  };
  return mfccCache;
}

function paddedHannWindow() {
  const window = new Float32Array(N_FFT);
  const start = Math.floor((N_FFT - WIN_LENGTH) / 2);
  for (let index = 0; index < WIN_LENGTH; index += 1) {
    window[start + index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / WIN_LENGTH);
  }
  return window;
}

function dftTable(fn) {
  const rows = [];
  for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
    const row = new Float32Array(N_FFT);
    for (let index = 0; index < N_FFT; index += 1) {
      row[index] = fn((2 * Math.PI * bin * index) / N_FFT);
    }
    rows.push(row);
  }
  return rows;
}

function melFilterbank() {
  const filters = Array.from({ length: N_MELS }, () => []);
  const fMin = 0;
  const fMax = MODEL_SAMPLE_RATE / 2;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const points = [];
  for (let index = 0; index < N_MELS + 2; index += 1) {
    const mel = melMin + ((melMax - melMin) * index) / (N_MELS + 1);
    points.push(melToHz(mel));
  }
  const fftFreqs = [];
  for (let bin = 0; bin <= N_FFT / 2; bin += 1) {
    fftFreqs.push((bin * MODEL_SAMPLE_RATE) / N_FFT);
  }
  for (let melIndex = 0; melIndex < N_MELS; melIndex += 1) {
    const lower = points[melIndex];
    const center = points[melIndex + 1];
    const upper = points[melIndex + 2];
    for (let bin = 0; bin < fftFreqs.length; bin += 1) {
      const freq = fftFreqs[bin];
      let weight = 0;
      if (freq >= lower && freq <= center) {
        weight = (freq - lower) / (center - lower);
      } else if (freq > center && freq <= upper) {
        weight = (upper - freq) / (upper - center);
      }
      if (weight > 0) filters[melIndex].push([bin, weight]);
    }
  }
  return filters;
}

function hzToMel(freq) {
  return 2595 * Math.log10(1 + freq / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function dctMatrix() {
  const matrix = [];
  for (let mel = 0; mel < N_MELS; mel += 1) {
    const row = new Float32Array(N_MFCC);
    for (let coeff = 0; coeff < N_MFCC; coeff += 1) {
      let value = Math.cos((Math.PI / N_MELS) * (mel + 0.5) * coeff);
      if (coeff === 0) value *= 1 / Math.sqrt(2);
      row[coeff] = value * Math.sqrt(2 / N_MELS);
    }
    matrix.push(row);
  }
  return matrix;
}

function reflectPad(samples, pad) {
  const result = new Float32Array(samples.length + pad * 2);
  for (let index = 0; index < pad; index += 1) {
    result[index] = samples[pad - index];
    result[pad + samples.length + index] = samples[samples.length - 2 - index];
  }
  result.set(samples, pad);
  return result;
}
