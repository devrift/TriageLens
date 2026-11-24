/**
 * TriageLens Clinical Benchmark Harness
 * 
 * Executes large-scale batch validation (1,000 to 10,000+ vectors) against the core DSP and vision pipelines.
 * Computes Pearson correlation (r), Mean Absolute Error (MAE), Root Mean Square Error (RMSE),
 * and Accuracy % according to ANSI/AAMI EC13 and ISO 80601-2-61 clinical vital sign monitoring standards.
 */

import {
  BiquadFilter,
  PeakDetector,
  PlaneOrthogonalToSkin,
  SignalQualityIndex,
  UniformResampler,
} from '../core/DSPFilters';
import { FaceMeshEngine } from '../core/FaceMeshEngine';
import { TriageCoordinator } from '../core/TriageCoordinator';
import {
  FitzpatrickType,
  SyntheticBenchmarkSample,
} from './PhysiomeDatasetGenerator';

export interface VitalMetricsSummary {
  sampleCount: number;
  pearsonR: number;
  mae: number;
  rmse: number;
  accuracyWithinTolerancePct: number; // e.g. within +- 3.0 BPM or +- 2.0 RPM
}

export interface BenchmarkReport {
  totalVectors: number;
  durationMs: number;
  heartRate: VitalMetricsSummary;
  respirationRate: VitalMetricsSummary;
  strokeAsymmetry: VitalMetricsSummary;
  fitzpatrickStratification: Record<FitzpatrickType, { hrMae: number; hrPearsonR: number; count: number }>;
  snrStratification: {
    lowSnr: { hrMae: number; count: number };    // < 6 dB
    midSnr: { hrMae: number; count: number };    // 6 - 15 dB
    highSnr: { hrMae: number; count: number };   // > 15 dB
  };
}

export class ClinicalBenchmarkHarness {
  /**
   * Calculate Pearson correlation coefficient r between two equal-length arrays
   */
  public static calculatePearsonR(x: number[], y: number[]): number {
    const n = x.length;
    if (n < 2) return 1.0;

    let sumX = 0.0;
    let sumY = 0.0;
    for (let i = 0; i < n; i++) {
      sumX += x[i];
      sumY += y[i];
    }
    const meanX = sumX / n;
    const meanY = sumY / n;

    let num = 0.0;
    let denX = 0.0;
    let denY = 0.0;

    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    const den = Math.sqrt(denX * denY);
    if (den < 1e-12) return 0.0;
    return num / den;
  }

  /**
   * Run benchmark on an array of synthetic samples
   */
  public static evaluateBatch(samples: SyntheticBenchmarkSample[]): BenchmarkReport {
    const tStart = typeof performance !== 'undefined' ? performance.now() : Date.now();

    const gtHr: number[] = [];
    const estHr: number[] = [];
    const gtRr: number[] = [];
    const estRr: number[] = [];
    const gtAsym: number[] = [];
    const estAsym: number[] = [];

    const fitzpatrickBuckets: Record<FitzpatrickType, { gt: number[]; est: number[] }> = {
      I: { gt: [], est: [] },
      II: { gt: [], est: [] },
      III: { gt: [], est: [] },
      IV: { gt: [], est: [] },
      V: { gt: [], est: [] },
      VI: { gt: [], est: [] },
    };

    const snrBuckets = {
      low: { gt: [] as number[], est: [] as number[] },
      mid: { gt: [] as number[], est: [] as number[] },
      high: { gt: [] as number[], est: [] as number[] },
    };

    // Pre-allocate DSP pipeline objects outside loop to prevent GC overhead
    const resamplerR = new UniformResampler(30.0);
    const resamplerG = new UniformResampler(30.0);
    const resamplerB = new UniformResampler(30.0);
    const pos = new PlaneOrthogonalToSkin();
    const filter = BiquadFilter.createButterworthBandpass30Hz();
    const peakDetector = new PeakDetector();
    const sqi = new SignalQualityIndex();
    const coordinator = new TriageCoordinator();

    const rawSamplesBuffer: number[] = [];

    for (let sIdx = 0; sIdx < samples.length; sIdx++) {
      const sample = samples[sIdx];
      const gt = sample.groundTruth;

      resamplerR.reset();
      resamplerG.reset();
      resamplerB.reset();
      pos.reset();
      filter.reset();
      peakDetector.reset();
      sqi.reset();
      rawSamplesBuffer.length = 0;

      const numFrames = sample.timestamps.length;

      for (let i = 0; i < numFrames; i++) {
        const ts = sample.timestamps[i];
        const r = sample.rChannel[i];
        const g = sample.gChannel[i];
        const b = sample.bChannel[i];

        resamplerR.push(ts, r, (uTs, uR) => {
          resamplerG.push(uTs, g, (_tsG, uG) => {
            resamplerB.push(uTs, b, (_tsB, uB) => {
              const h = pos.process(uR, uG, uB);
              rawSamplesBuffer.push(h);
              const filtered = filter.process(h);
              sqi.push(filtered);
              peakDetector.process(uTs, filtered);
            });
          });
        });
      }

      // Evaluate Heart Rate via Intelligent Dual-Domain Clinical Fusion
      const peakBpm = peakDetector.getCurrentBpm();
      const sqiEval = sqi.evaluate(30.0);
      const fftBpm = sqiEval.peakFreqHz * 60.0;

      let estimatedBpm = peakBpm;
      if (fftBpm >= 38.0 && fftBpm <= 200.0) {
        if (peakBpm < 35.0 || peakBpm > 220.0) {
          estimatedBpm = fftBpm;
        } else {
          const delta = Math.abs(peakBpm - fftBpm);
          if (delta <= 4.0) {
            estimatedBpm = 0.50 * peakBpm + 0.50 * fftBpm;
          } else {
            if (sqiEval.sqi >= 0.30) {
              estimatedBpm = fftBpm;
            } else {
              estimatedBpm = 0.50 * peakBpm + 0.50 * fftBpm;
            }
          }
        }
      }

      // Evaluate Respiration Rate (using raw un-bandpassed rPPG signal)
      const estimatedRpm = coordinator.estimateRespirationRate(rawSamplesBuffer, 30.0);

      // Evaluate Facial Symmetry (projecting 3D synthetic landmark landmarks)
      let estimatedAsym = 0.0;
      if (sample.facialLandmarks3D.length > 0) {
        const frame0 = sample.facialLandmarks3D[0];
        estimatedAsym = FaceMeshEngine.calculateStrokeAsymmetry(frame0);
      }

      gtHr.push(gt.heartRateBpm);
      estHr.push(estimatedBpm);
      gtRr.push(gt.respirationRateRpm);
      estRr.push(estimatedRpm);
      gtAsym.push(gt.strokeAsymmetryDelta);
      estAsym.push(estimatedAsym);

      // Bucketing
      fitzpatrickBuckets[gt.fitzpatrick].gt.push(gt.heartRateBpm);
      fitzpatrickBuckets[gt.fitzpatrick].est.push(estimatedBpm);

      if (gt.snrDb < 6.0) {
        snrBuckets.low.gt.push(gt.heartRateBpm);
        snrBuckets.low.est.push(estimatedBpm);
      } else if (gt.snrDb <= 15.0) {
        snrBuckets.mid.gt.push(gt.heartRateBpm);
        snrBuckets.mid.est.push(estimatedBpm);
      } else {
        snrBuckets.high.gt.push(gt.heartRateBpm);
        snrBuckets.high.est.push(estimatedBpm);
      }
    }

    const computeSummary = (gt: number[], est: number[], tolerance: number): VitalMetricsSummary => {
      const n = gt.length;
      let absErrorSum = 0.0;
      let sqErrorSum = 0.0;
      let withinToleranceCount = 0;

      for (let i = 0; i < n; i++) {
        const err = Math.abs(est[i] - gt[i]);
        absErrorSum += err;
        sqErrorSum += err * err;
        if (err <= tolerance) {
          withinToleranceCount++;
        }
      }

      return {
        sampleCount: n,
        pearsonR: ClinicalBenchmarkHarness.calculatePearsonR(gt, est),
        mae: n > 0 ? absErrorSum / n : 0.0,
        rmse: n > 0 ? Math.sqrt(sqErrorSum / n) : 0.0,
        accuracyWithinTolerancePct: n > 0 ? (withinToleranceCount / n) * 100.0 : 0.0,
      };
    };

    const hrSummary = computeSummary(gtHr, estHr, 3.0); // +- 3 BPM
    const rrSummary = computeSummary(gtRr, estRr, 2.0); // +- 2 RPM
    const asymSummary = computeSummary(gtAsym, estAsym, 0.03); // +- 0.03 Delta

    const fitzpatrickStrat: Record<FitzpatrickType, { hrMae: number; hrPearsonR: number; count: number }> = {
      I: { hrMae: 0, hrPearsonR: 0, count: 0 },
      II: { hrMae: 0, hrPearsonR: 0, count: 0 },
      III: { hrMae: 0, hrPearsonR: 0, count: 0 },
      IV: { hrMae: 0, hrPearsonR: 0, count: 0 },
      V: { hrMae: 0, hrPearsonR: 0, count: 0 },
      VI: { hrMae: 0, hrPearsonR: 0, count: 0 },
    };

    for (const key of ['I', 'II', 'III', 'IV', 'V', 'VI'] as FitzpatrickType[]) {
      const b = fitzpatrickBuckets[key];
      const count = b.gt.length;
      let mae = 0;
      for (let i = 0; i < count; i++) mae += Math.abs(b.est[i] - b.gt[i]);
      fitzpatrickStrat[key] = {
        count,
        hrMae: count > 0 ? mae / count : 0,
        hrPearsonR: ClinicalBenchmarkHarness.calculatePearsonR(b.gt, b.est),
      };
    }

    const calcBucketMae = (b: { gt: number[]; est: number[] }) => {
      let mae = 0;
      for (let i = 0; i < b.gt.length; i++) mae += Math.abs(b.est[i] - b.gt[i]);
      return { hrMae: b.gt.length > 0 ? mae / b.gt.length : 0, count: b.gt.length };
    };

    const tEnd = typeof performance !== 'undefined' ? performance.now() : Date.now();

    return {
      totalVectors: samples.length,
      durationMs: tEnd - tStart,
      heartRate: hrSummary,
      respirationRate: rrSummary,
      strokeAsymmetry: asymSummary,
      fitzpatrickStratification: fitzpatrickStrat,
      snrStratification: {
        lowSnr: calcBucketMae(snrBuckets.low),
        midSnr: calcBucketMae(snrBuckets.mid),
        highSnr: calcBucketMae(snrBuckets.high),
      },
    };
  }
}
