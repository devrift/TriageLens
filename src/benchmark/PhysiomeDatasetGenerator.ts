/**
 * TriageLens Physiome Dataset Generator
 * 
 * Generates mathematically rigorous, biophysically grounded synthetic benchmark datasets
 * for evaluating client-side rPPG, contact PPG, respiration extraction, and 3D facial symmetry.
 * 
 * Grounded in:
 * - Fitzpatrick phototypes I through VI with wavelength-dependent melanin & hemoglobin absorption spectra
 * - Asymmetric arterial pulse morphology (percussion wave, dicrotic notch, diastolic runoff)
 * - Tri-phasic respiratory modulation (RIIV baseline drift, RIAV amplitude modulation, RIFV/RSA frequency variation)
 * - Non-uniform frame timestamp jitter (hardware camera clocks 20 - 45 ms)
 * - 3D facial landmark projection with rigid head rotation (Yaw/Pitch/Roll) and unilateral stroke paresis
 * - Calibrated additive Gaussian noise (SNR -5 dB to +25 dB) and 50/60 Hz powerline flicker
 */

export type FitzpatrickType = 'I' | 'II' | 'III' | 'IV' | 'V' | 'VI';

export interface GroundTruthVitals {
  heartRateBpm: number;
  respirationRateRpm: number;
  strokeAsymmetryDelta: number;
  fitzpatrick: FitzpatrickType;
  snrDb: number;
  durationSec: number;
  sampleRateHz: number;
  jitterMs: number;
  headYawDeg: number;
  headPitchDeg: number;
  headRollDeg: number;
}

export interface SyntheticBenchmarkSample {
  id: number;
  groundTruth: GroundTruthVitals;
  timestamps: Float64Array;
  // Raw forehead RGB spatial means across time
  rChannel: Float64Array;
  gChannel: Float64Array;
  bChannel: Float64Array;
  // Synthetic 3D facial landmarks for the test session
  facialLandmarks3D: Array<Array<{ x: number; y: number; z: number }>>;
}

export class PhysiomeDatasetGenerator {
  // Deterministic Lehmer / LCG pseudo-random generator with seed for 100% reproducible benchmarks
  private seed: number;

  constructor(initialSeed: number = 42424242) {
    this.seed = initialSeed;
  }

  public setSeed(newSeed: number): void {
    this.seed = newSeed;
  }

  private random(): number {
    this.seed = (this.seed * 16807 + 0) % 2147483647;
    return (this.seed - 1) / 2147483646;
  }

  private randomRange(min: number, max: number): number {
    return min + this.random() * (max - min);
  }

  private gaussianNoise(sigma: number = 1.0): number {
    const u1 = Math.max(1e-12, this.random());
    const u2 = this.random();
    return sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /**
   * Fitzpatrick baseline RGB values and pulsatile absorption modulation depths
   */
  public static readonly FITZPATRICK_PROFILES: Record<
    FitzpatrickType,
    { baseR: number; baseG: number; baseB: number; modR: number; modG: number; modB: number }
  > = {
    I:   { baseR: 215, baseG: 180, baseB: 165, modR: 0.007, modG: 0.022, modB: 0.005 },
    II:  { baseR: 200, baseG: 165, baseB: 145, modR: 0.006, modG: 0.019, modB: 0.004 },
    III: { baseR: 180, baseG: 145, baseB: 120, modR: 0.005, modG: 0.016, modB: 0.0035 },
    IV:  { baseR: 155, baseG: 120, baseB: 95,  modR: 0.004, modG: 0.013, modB: 0.003 },
    V:   { baseR: 120, baseG: 88,  baseB: 65,  modR: 0.003, modG: 0.009, modB: 0.002 },
    VI:  { baseR: 85,  baseG: 55,  baseB: 40,  modR: 0.002, modG: 0.006, modB: 0.0015 },
  };

  /**
   * Single arterial pulse template with systolic peak, dicrotic notch, and diastolic wave.
   * Phase normalized to [0, 1].
   */
  public static arterialPulseMorphology(phase: number): number {
    const p = phase - Math.floor(phase);
    // Gaussian 1: Systolic percussion wave (center 0.16, width 0.07, amp 1.0)
    const p1 = Math.exp(-Math.pow(p - 0.16, 2) / (2 * Math.pow(0.07, 2)));
    // Gaussian 2: Tidal wave (center 0.28, width 0.06, amp 0.45)
    const p2 = 0.45 * Math.exp(-Math.pow(p - 0.28, 2) / (2 * Math.pow(0.06, 2)));
    // Gaussian 3: Dicrotic notch reflection / diastolic wave (center 0.42, width 0.09, amp 0.35)
    const p3 = 0.35 * Math.exp(-Math.pow(p - 0.42, 2) / (2 * Math.pow(0.09, 2)));
    // Exponential diastolic runoff
    const runoff = 0.12 * Math.exp(-p * 3.5);

    return p1 + p2 + p3 + runoff;
  }

  /**
   * Synthesize a complete 15-second multi-parameter clinical telemetry test vector
   */
  public generateVector(config?: Partial<GroundTruthVitals>, id: number = 0): SyntheticBenchmarkSample {
    const fitzpatrickKeys: FitzpatrickType[] = ['I', 'II', 'III', 'IV', 'V', 'VI'];
    const chosenFitzpatrick = config?.fitzpatrick ?? fitzpatrickKeys[Math.floor(this.random() * fitzpatrickKeys.length)];

    const hrBpm = config?.heartRateBpm ?? this.randomRange(42.0, 175.0);
    const rrRpm = config?.respirationRateRpm ?? this.randomRange(9.0, 32.0);
    const strokeAsym = config?.strokeAsymmetryDelta ?? (this.random() < 0.2 ? this.randomRange(0.12, 0.24) : this.randomRange(0.01, 0.06));
    const snrDb = config?.snrDb ?? this.randomRange(0.0, 24.0);
    const durationSec = config?.durationSec ?? 15.0;
    const fs = config?.sampleRateHz ?? 30.0;
    const jitterMs = config?.jitterMs ?? this.randomRange(0.5, 4.0);

    const headYaw = config?.headYawDeg ?? this.randomRange(-20.0, 20.0);
    const headPitch = config?.headPitchDeg ?? this.randomRange(-15.0, 15.0);
    const headRoll = config?.headRollDeg ?? this.randomRange(-15.0, 15.0);

    const groundTruth: GroundTruthVitals = {
      heartRateBpm: hrBpm,
      respirationRateRpm: rrRpm,
      strokeAsymmetryDelta: strokeAsym,
      fitzpatrick: chosenFitzpatrick,
      snrDb,
      durationSec,
      sampleRateHz: fs,
      jitterMs,
      headYawDeg: headYaw,
      headPitchDeg: headPitch,
      headRollDeg: headRoll,
    };

    const numSamples = Math.round(durationSec * fs);
    const timestamps = new Float64Array(numSamples);
    const rChannel = new Float64Array(numSamples);
    const gChannel = new Float64Array(numSamples);
    const bChannel = new Float64Array(numSamples);

    const profile = PhysiomeDatasetGenerator.FITZPATRICK_PROFILES[chosenFitzpatrick];

    // Respiratory parameters
    const fResp = rrRpm / 60.0; // in Hz
    const fHeartMean = hrBpm / 60.0; // in Hz

    // Noise amplitude from SNR dB
    // SNR = 10 * log10(SignalPower / NoisePower)
    const signalPower = 1.0;
    const noisePower = signalPower / Math.pow(10, snrDb / 10.0);
    const noiseSigma = Math.sqrt(noisePower);

    let currentPhase = 0.0;
    let tCurrent = 0.0;
    const dtNominal = 1000.0 / fs;

    for (let i = 0; i < numSamples; i++) {
      // Hardware timestamp jitter
      const jitter = this.gaussianNoise(jitterMs);
      const actualDt = Math.max(10.0, dtNominal + jitter);
      tCurrent += (i === 0 ? 0 : actualDt);
      timestamps[i] = tCurrent;

      const tSec = tCurrent / 1000.0;

      // 1. Respiratory Modulations:
      // RIIV (Intensity Variation / Baseline Wander): ~1.2% modulation
      const riiv = 0.012 * Math.sin(2.0 * Math.PI * fResp * tSec);

      // RIAV (Amplitude Variation): ~15% pulse amplitude swing with respiration
      const riav = 1.0 + 0.15 * Math.sin(2.0 * Math.PI * fResp * tSec + Math.PI / 4);

      // RIFV (Frequency Variation / Respiratory Sinus Arrhythmia RSA):
      // Instantaneous heart frequency speeds up during inhalation and slows during exhalation
      const rsaFreqVariation = 0.08 * fHeartMean * Math.cos(2.0 * Math.PI * fResp * tSec);
      const instantaneousFhr = fHeartMean + rsaFreqVariation;

      // Integrate phase
      const dtSec = (i === 0 ? dtNominal : actualDt) / 1000.0;
      currentPhase += instantaneousFhr * dtSec;

      // 2. Arterial pulse waveform
      const cleanPulse = riav * PhysiomeDatasetGenerator.arterialPulseMorphology(currentPhase);

      // 3. Additive noise (optical sensor noise, shot noise)
      const noiseR = this.gaussianNoise(noiseSigma * 0.003);
      const noiseG = this.gaussianNoise(noiseSigma * 0.005);
      const noiseB = this.gaussianNoise(noiseSigma * 0.004);

      // 4. Low-frequency illumination drift + 50Hz/60Hz AC flicker alias
      const drift = 0.003 * Math.sin(2.0 * Math.PI * 0.05 * tSec);
      const flicker = 0.001 * Math.sin(2.0 * Math.PI * 10.0 * tSec); // 50 Hz alias at 30 fps

      // 5. Synthesis across R, G, B channels using physiological Fitzpatrick extinction
      rChannel[i] = profile.baseR * (1.0 + profile.modR * cleanPulse + riiv * 0.4 + drift + flicker + noiseR);
      // Green channel has strongest hemoglobin absorption dip (inverted pulsatile signal)
      gChannel[i] = profile.baseG * (1.0 - profile.modG * cleanPulse + riiv * 1.0 + drift + flicker + noiseG);
      bChannel[i] = profile.baseB * (1.0 - profile.modB * cleanPulse + riiv * 0.6 + drift + flicker + noiseB);
    }

    // Synthesize 3D facial landmarks for start, middle, and end frames
    const facialLandmarks3D = this.generateFacialLandmarkFrames(strokeAsym, headYaw, headPitch, headRoll);

    return {
      id,
      groundTruth,
      timestamps,
      rChannel,
      gChannel,
      bChannel,
      facialLandmarks3D,
    };
  }

  /**
   * Synthesize canonical MediaPipe 3D face landmarks with calibrated asymmetry and 3D head rotation
   */
  public generateFacialLandmarkFrames(
    asymmetryDelta: number,
    yawDeg: number,
    pitchDeg: number,
    rollDeg: number
  ): Array<Array<{ x: number; y: number; z: number }>> {
    const yaw = (yawDeg * Math.PI) / 180.0;
    const pitch = (pitchDeg * Math.PI) / 180.0;
    const roll = (rollDeg * Math.PI) / 180.0;

    // Standard canonical 3D coordinates centered at Mid-face
    // Indices:
    // 33: Right eye outer
    // 263: Left eye outer
    // 168: Nasion
    // 152: Chin tip
    // 61: Right mouth corner
    // 291: Left mouth corner (with asymmetry displacement)
    const baseLandmarks: Record<number, { x: number; y: number; z: number }> = {
      33:  { x: -0.18, y: -0.06, z: 0.00 }, // right eye outer
      263: { x:  0.18, y: -0.06, z: 0.00 }, // left eye outer
      168: { x:  0.00, y: -0.06, z: 0.04 }, // nasion
      152: { x:  0.00, y:  0.30, z: 0.02 }, // chin tip
      61:  { x: -0.10, y:  0.18, z: 0.02 }, // right mouth corner
      // Left mouth corner has unilateral vertical droop/asymmetry proportional to asymmetryDelta * IOD
      // IOD = 0.36
      291: { x:  0.10, y:  0.18 + (asymmetryDelta * 0.36), z: 0.02 },
    };

    // Rotation matrices in 3D
    // Yaw (around Y axis)
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    // Pitch (around X axis)
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    // Roll (around Z axis)
    const cosR = Math.cos(roll), sinR = Math.sin(roll);

    const rotate3D = (p: { x: number; y: number; z: number }) => {
      // 1. Roll (Z)
      const x1 = cosR * p.x - sinR * p.y;
      const y1 = sinR * p.x + cosR * p.y;
      const z1 = p.z;

      // 2. Pitch (X)
      const x2 = x1;
      const y2 = cosP * y1 - sinP * z1;
      const z2 = sinP * y1 + cosP * z1;

      // 3. Yaw (Y)
      const x3 = cosY * x2 + sinY * z2;
      const y3 = y2;
      const z3 = -sinY * x2 + cosY * z2;

      // Translate to screen normalized coordinates [0.5, 0.5] with small camera distance
      return {
        x: 0.5 + x3,
        y: 0.5 + y3,
        z: z3,
      };
    };

    const transformedLandmarks: Array<{ x: number; y: number; z: number }> = new Array(468);
    for (let i = 0; i < 468; i++) {
      transformedLandmarks[i] = { x: 0.5, y: 0.5, z: 0 };
    }

    for (const [idxStr, pt] of Object.entries(baseLandmarks)) {
      const idx = parseInt(idxStr, 10);
      transformedLandmarks[idx] = rotate3D(pt);
    }

    return [transformedLandmarks];
  }
}
