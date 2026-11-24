import { describe, it, expect } from 'vitest';
import {
  BiquadFilter,
  PlaneOrthogonalToSkin,
  PeakDetector,
  SignalQualityIndex,
  UniformResampler,
} from '../DSPFilters';

describe('DSPFilters Test Suite', () => {
  describe('1. Filter Stability', () => {
    it('maintains zero-input stability with no NaN or numeric divergence from cold start', () => {
      const cascade = BiquadFilter.createButterworthBandpass30Hz();
      for (let i = 0; i < 300; i++) {
        const out = cascade.process(0.0);
        expect(Number.isFinite(out)).toBe(true);
        expect(Number.isNaN(out)).toBe(false);
        expect(out).toBe(0.0);
      }
    });

    it('settles to zero without NaN or numeric divergence when fed 300 zero samples after excitation', () => {
      const cascade = BiquadFilter.createButterworthBandpass30Hz();

      // Introduce an impulse to excite filter registers
      cascade.process(1.0);

      // Now feed 300 zero samples
      let finalSample = 0.0;
      for (let i = 0; i < 300; i++) {
        finalSample = cascade.process(0.0);
        expect(Number.isFinite(finalSample)).toBe(true);
        expect(Number.isNaN(finalSample)).toBe(false);
      }

      // Verify decay to below noise threshold
      expect(Math.abs(finalSample)).toBeLessThan(1e-6);
    });

    it('recovers gracefully from non-finite input without state corruption', () => {
      const cascade = BiquadFilter.createButterworthBandpass30Hz();
      const outNaN = cascade.process(NaN);
      expect(Number.isFinite(outNaN)).toBe(true);
      const outInf = cascade.process(Infinity);
      expect(Number.isFinite(outInf)).toBe(true);
    });
  });

  describe('2. Frequency Selectivity', () => {
    const fs = 30.0;
    const cascade = BiquadFilter.createButterworthBandpass30Hz();

    function measureFilterGain(freqHz: number, durationSec: number = 10): { gain: number; gainDb: number } {
      cascade.reset();
      const numSamples = Math.round(durationSec * fs);
      const inputAmplitude = 1.0;

      // Settle transient response
      const settleSamples = Math.round(4.0 * fs);
      let maxOutput = 0.0;

      for (let i = 0; i < numSamples; i++) {
        const t = i / fs;
        const x = inputAmplitude * Math.sin(2.0 * Math.PI * freqHz * t);
        const y = cascade.process(x);

        if (i >= settleSamples) {
          const absY = Math.abs(y);
          if (absY > maxOutput) {
            maxOutput = absY;
          }
        }
      }

      const gain = maxOutput / inputAmplitude;
      const gainDb = 20.0 * Math.log10(gain);
      return { gain, gainDb };
    }

    it('passes 1.2 Hz sine wave (72 BPM cardiac) with < 1 dB attenuation', () => {
      const { gainDb } = measureFilterGain(1.2);
      const attenuationDb = -gainDb;
      expect(attenuationDb).toBeLessThan(1.0);
    });

    it('attenuates 0.1 Hz sine wave (baseline drift) by > 18 dB', () => {
      const { gainDb } = measureFilterGain(0.1, 15);
      const attenuationDb = -gainDb;
      expect(attenuationDb).toBeGreaterThan(18.0);
    });

    it('attenuates 6.0 Hz sine wave (lighting flicker) by > 18 dB', () => {
      const { gainDb } = measureFilterGain(6.0, 10);
      const attenuationDb = -gainDb;
      expect(attenuationDb).toBeGreaterThan(18.0);
    });
  });

  describe('3. Peak Accuracy Across Heart Rates', () => {
    // Deterministic Box-Muller pseudo-random generator
    function createNoiseGenerator() {
      let seed = 1234567;
      function pseudoRandom(): number {
        seed = (seed * 16807 + 0) % 2147483647;
        return (seed - 1) / 2147483646;
      }
      return function gaussianNoise(sigma: number = 0.05): number {
        const u1 = Math.max(1e-9, pseudoRandom());
        const u2 = pseudoRandom();
        return sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      };
    }

    it('extracts pulse rate within +- 1 BPM from synthetic 75 BPM pulse train with Gaussian noise', () => {
      const detector = new PeakDetector();
      const fs = 30.0;
      const targetBpm = 75.0;
      const durationSec = 15.0;
      const totalSamples = Math.round(durationSec * fs);
      const noise = createNoiseGenerator();

      let detectedCount = 0;
      let lastReportedBpm = 0;

      for (let i = 0; i < totalSamples; i++) {
        const timestamp = i * (1000.0 / fs);
        const phase = (2.0 * Math.PI * (targetBpm / 60.0) * timestamp) / 1000.0;
        const pulse = Math.cos(phase) + 0.3 * Math.cos(2 * phase);
        const noisySample = pulse + noise(0.05);

        const peak = detector.process(timestamp, noisySample);
        if (peak) {
          detectedCount++;
          lastReportedBpm = peak.bpm;
        }
      }

      expect(detectedCount).toBeGreaterThan(10);
      expect(Math.abs(lastReportedBpm - targetBpm)).toBeLessThanOrEqual(1.0);
    });

    it('extracts low heart rate at 50 BPM (bradycardia baseline)', () => {
      const detector = new PeakDetector();
      const fs = 30.0;
      const targetBpm = 50.0;
      const durationSec = 15.0;
      const totalSamples = Math.round(durationSec * fs);

      let lastReportedBpm = 0;
      for (let i = 0; i < totalSamples; i++) {
        const timestamp = i * (1000.0 / fs);
        const phase = (2.0 * Math.PI * (targetBpm / 60.0) * timestamp) / 1000.0;
        const pulse = Math.cos(phase);
        const peak = detector.process(timestamp, pulse);
        if (peak) {
          lastReportedBpm = peak.bpm;
        }
      }

      expect(Math.abs(lastReportedBpm - targetBpm)).toBeLessThanOrEqual(2.0);
    });

    it('extracts high heart rate at 150 BPM (tachycardia, >= 300ms lockout)', () => {
      const detector = new PeakDetector();
      const fs = 30.0;
      const targetBpm = 150.0; // 400ms interval (exceeds 300ms refractory)
      const durationSec = 15.0;
      const totalSamples = Math.round(durationSec * fs);

      let lastReportedBpm = 0;
      for (let i = 0; i < totalSamples; i++) {
        const timestamp = i * (1000.0 / fs);
        const phase = (2.0 * Math.PI * (targetBpm / 60.0) * timestamp) / 1000.0;
        const pulse = Math.cos(phase);
        const peak = detector.process(timestamp, pulse);
        if (peak) {
          lastReportedBpm = peak.bpm;
        }
      }

      expect(Math.abs(lastReportedBpm - targetBpm)).toBeLessThanOrEqual(2.0);
    });
  });

  describe('Plane-Orthogonal-to-Skin (POS)', () => {
    it('computes pulsatile blood volume signal without NaN or divergence', () => {
      const pos = new PlaneOrthogonalToSkin();
      const fs = 30.0;
      const durationSec = 3.0;
      const totalSamples = Math.round(durationSec * fs);

      for (let i = 0; i < totalSamples; i++) {
        const t = i / fs;
        const r = 180 + 2.0 * Math.sin(2.0 * Math.PI * 1.2 * t);
        const g = 120 - 4.0 * Math.sin(2.0 * Math.PI * 1.2 * t);
        const b = 100 + 0.5 * Math.sin(2.0 * Math.PI * 1.2 * t);

        const h = pos.process(r, g, b);
        expect(Number.isFinite(h)).toBe(true);
        expect(Number.isNaN(h)).toBe(false);
      }
    });

    it('handles flat-RGB zero-variance input safely without division by zero', () => {
      const pos = new PlaneOrthogonalToSkin();
      for (let i = 0; i < 60; i++) {
        const h = pos.process(150, 150, 150);
        expect(Number.isFinite(h)).toBe(true);
        expect(Number.isNaN(h)).toBe(false);
      }
    });

    it('handles zero luminance input safely', () => {
      const pos = new PlaneOrthogonalToSkin();
      for (let i = 0; i < 60; i++) {
        const h = pos.process(0, 0, 0);
        expect(h).toBe(0.0);
      }
    });

    it('absorbs sudden lighting step change (+50 units) without numeric explosion', () => {
      const pos = new PlaneOrthogonalToSkin();
      for (let i = 0; i < 45; i++) {
        pos.process(100, 100, 100);
      }
      // Step jump
      const hJump = pos.process(150, 150, 150);
      expect(Number.isFinite(hJump)).toBe(true);
    });
  });

  describe('UniformResampler', () => {
    it('interpolates jittered hardware frames onto a uniform 30 Hz grid', () => {
      const resampler = new UniformResampler(30.0);
      const uniformOutputs: { ts: number; val: number }[] = [];

      let t = 1000.0;
      for (let i = 0; i < 100; i++) {
        const jitter = (i % 5) * 2 - 4; // -4 to +4 ms jitter
        t += 33.333 + jitter;
        const val = Math.sin((2.0 * Math.PI * 1.0 * t) / 1000.0);

        resampler.push(t, val, (uTs, uVal) => {
          uniformOutputs.push({ ts: uTs, val: uVal });
        });
      }

      expect(uniformOutputs.length).toBeGreaterThan(80);

      const dtExpected = 1000.0 / 30.0;
      for (let i = 1; i < uniformOutputs.length; i++) {
        const dt = uniformOutputs[i].ts - uniformOutputs[i - 1].ts;
        expect(Math.abs(dt - dtExpected)).toBeLessThan(1e-4);
      }
    });

    it('gracefully handles duplicate and out-of-order timestamps', () => {
      const resampler = new UniformResampler(30.0);
      const uniformOutputs: { ts: number; val: number }[] = [];

      resampler.push(1000, 1.0, (uTs, uVal) => uniformOutputs.push({ ts: uTs, val: uVal }));
      // Duplicate timestamp
      resampler.push(1000, 1.5, (uTs, uVal) => uniformOutputs.push({ ts: uTs, val: uVal }));
      // Out-of-order backwards timestamp
      resampler.push(950, 0.5, (uTs, uVal) => uniformOutputs.push({ ts: uTs, val: uVal }));
      // Valid advance
      resampler.push(1040, 2.0, (uTs, uVal) => uniformOutputs.push({ ts: uTs, val: uVal }));

      expect(uniformOutputs.length).toBeGreaterThanOrEqual(1);
    });

    it('recovers cleanly from long camera pause gaps (>500ms)', () => {
      const resampler = new UniformResampler(30.0);
      let count = 0;

      resampler.push(1000, 1.0, () => count++);
      resampler.push(1035, 1.1, () => count++);
      // 2-second pause gap
      resampler.push(3035, 1.2, () => count++);
      resampler.push(3070, 1.3, () => count++);

      expect(count).toBeLessThan(10); // Does not generate 60 runaway intermediate frames
    });
  });

  describe('SignalQualityIndex (SQI)', () => {
    it('evaluates high SQI (>0.80) on clean 1.2 Hz cardiac signal', () => {
      const sqiEngine = new SignalQualityIndex();
      const fs = 30.0;

      for (let i = 0; i < 256; i++) {
        const t = i / fs;
        const s = Math.sin(2 * Math.PI * 1.2 * t) + 0.25 * Math.sin(2 * Math.PI * 2.4 * t);
        sqiEngine.push(s);
      }

      const result = sqiEngine.evaluate(fs);
      expect(result.sqi).toBeGreaterThan(0.80);
      expect(result.snrDb).toBeGreaterThan(6.0);
      expect(Math.abs(result.peakFreqHz - 1.2)).toBeLessThan(0.2);
    });

    it('evaluates low SQI (<0.40) on broadband random noise', () => {
      const sqiEngine = new SignalQualityIndex();
      let seed = 987654321;
      function rand(): number {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff - 0.5;
      }

      for (let i = 0; i < 256; i++) {
        sqiEngine.push(rand());
      }

      const result = sqiEngine.evaluate(30.0);
      expect(result.sqi).toBeLessThan(0.40);
    });

    it('returns SQI = 0.0 on flat buffer without throwing', () => {
      const sqiEngine = new SignalQualityIndex();
      for (let i = 0; i < 256; i++) {
        sqiEngine.push(0.0);
      }
      const result = sqiEngine.evaluate(30.0);
      expect(result.sqi).toBe(0.0);
      expect(result.snrDb).toBe(-99.0);
    });
  });

  describe('4. Asymmetry Invariance (FAST Protocol)', () => {
    function createSyntheticLandmarks(): { x: number; y: number; z: number }[] {
      const landmarks: { x: number; y: number; z: number }[] = [];
      for (let i = 0; i < 468; i++) {
        landmarks.push({ x: 0.5, y: 0.5, z: 0.0 });
      }

      // Nasion (168) and Chin Tip (152) define midline
      landmarks[168] = { x: 0.5, y: 0.35, z: 0.0 };
      landmarks[152] = { x: 0.5, y: 0.85, z: 0.0 };

      // Outer eye corners: right (33), left (263) -> d_IOD = 0.20
      landmarks[33] = { x: 0.4, y: 0.35, z: 0.0 };
      landmarks[263] = { x: 0.6, y: 0.35, z: 0.0 };

      // Symmetric mouth corners: right (61), left (291)
      landmarks[61] = { x: 0.44, y: 0.65, z: 0.0 };
      landmarks[291] = { x: 0.56, y: 0.65, z: 0.0 };

      return landmarks;
    }

    it('asserts delta_asymmetry ~ 0.0 on symmetric landmark coordinates', async () => {
      const { FaceMeshEngine } = await import('../FaceMeshEngine');
      const landmarks = createSyntheticLandmarks();

      const delta = FaceMeshEngine.calculateStrokeAsymmetry(landmarks);
      expect(delta).toBeLessThan(0.01);
    });

    it('asserts delta_asymmetry >= 0.12 when introducing a 15% downward translation on Landmark 291', async () => {
      const { FaceMeshEngine } = await import('../FaceMeshEngine');
      const landmarks = createSyntheticLandmarks();

      // d_IOD = 0.20 => 15% translation = 0.03
      landmarks[291].y += 0.03;

      const delta = FaceMeshEngine.calculateStrokeAsymmetry(landmarks);
      expect(delta).toBeGreaterThanOrEqual(0.12);
      expect(delta).toBeCloseTo(0.15, 2);
    });
  });
});
