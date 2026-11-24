/**
 * TriageLens 10,000 Raw Video Frame Stream Evaluation
 * 
 * Streams 10,000 full 2D video frames (320x240 RGBA) across 22 patient sessions
 * directly into the optical computer vision and DSP pipeline.
 * Computes Pearson correlation (r), MAE, and RMSE for continuous validation.
 */

import {
  BiquadFilter,
  PeakDetector,
  PlaneOrthogonalToSkin,
  SignalQualityIndex,
  UniformResampler,
} from '../core/DSPFilters';
import { TriageCoordinator } from '../core/TriageCoordinator';
import {
  VideoFrameStreamDataset,
  FrameStreamTelemetryTarget,
  VideoFrame,
} from './VideoFrameStreamDataset';

export interface VideoEvaluationSummary {
  totalFrames: number;
  totalSessions: number;
  heartRatePearsonR: number;
  heartRateMae: number;
  respirationPearsonR: number;
  respirationMae: number;
  durationMs: number;
}

export function run10000VideoFrameEvaluation(seed: number = 33445566): VideoEvaluationSummary {
  console.log(`\n===============================================================`);
  console.log(`  TRIAGELENS 10,000 VIDEO FRAME EXTRACTION & DSP BENCHMARK    `);
  console.log(`===============================================================\n`);
  console.log(`Streaming 10,000 raw 2D camera frames (320x240 @ 30 FPS) through vision pipeline...`);

  const dataset = new VideoFrameStreamDataset(seed);
  const coordinator = new TriageCoordinator();

  const fitzTypes: Array<'I' | 'II' | 'III' | 'IV' | 'V' | 'VI'> = ['I', 'II', 'III', 'IV', 'V', 'VI'];
  const sessionTargets: FrameStreamTelemetryTarget[] = [];

  // 22 patient sessions of 455 frames each = 10,010 frames
  for (let s = 0; s < 22; s++) {
    const isFace = s < 18; // 18 face rPPG sessions, 4 finger contact PPG sessions
    const hr = 45.0 + (s * 5.8) % 130.0; // 45 to 175 BPM
    const rr = 10.0 + (s * 1.1) % 24.0; // 10 to 34 RPM
    sessionTargets.push({
      sessionId: s + 1,
      mode: isFace ? 'face' : 'finger',
      groundTruthHr: Math.round(hr * 10) / 10,
      groundTruthRr: Math.round(rr * 10) / 10,
      groundTruthAsym: 0.035,
      fitzpatrick: fitzTypes[s % fitzTypes.length],
      snrDb: 6.0 + (s * 3.2) % 18.0,
      frameCount: 455,
    });
  }

  const resamplerR = new UniformResampler(30.0);
  const resamplerG = new UniformResampler(30.0);
  const resamplerB = new UniformResampler(30.0);
  const pos = new PlaneOrthogonalToSkin();
  const filter = BiquadFilter.createButterworthBandpass30Hz();
  const peakDetector = new PeakDetector();
  const sqi = new SignalQualityIndex();
  const rawPpgBuffer: number[] = [];

  const gtHrList: number[] = [];
  const estHrList: number[] = [];
  const gtRrList: number[] = [];
  const estRrList: number[] = [];

  let totalFramesProcessed = 0;
  const tStart = Date.now();

  for (let sIdx = 0; sIdx < sessionTargets.length; sIdx++) {
    const target = sessionTargets[sIdx];

    resamplerR.reset();
    resamplerG.reset();
    resamplerB.reset();
    pos.reset();
    filter.reset();
    peakDetector.reset();
    sqi.reset();
    rawPpgBuffer.length = 0;

    dataset.generateSessionFrames(target, (frame: VideoFrame, _frameIdx: number) => {
      totalFramesProcessed++;

      // Optical ROI extraction from frame buffer:
      // Sample forehead patch (center 24% width, 14% height)
      const data = frame.data;
      const w = frame.width;
      const h = frame.height;

      let rMean = 0;
      let gMean = 0;
      let bMean = 0;

      if (target.mode === 'face') {
        const roiX0 = Math.round(w * 0.38);
        const roiX1 = Math.round(w * 0.62);
        const roiY0 = Math.round(h * 0.22);
        const roiY1 = Math.round(h * 0.36);

        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        for (let y = roiY0; y < roiY1; y++) {
          const rowOffset = y * w * 4;
          for (let x = roiX0; x < roiX1; x++) {
            const ptr = rowOffset + x * 4;
            sumR += data[ptr];
            sumG += data[ptr + 1];
            sumB += data[ptr + 2];
            count++;
          }
        }
        rMean = sumR / count;
        gMean = sumG / count;
        bMean = sumB / count;
      } else {
        // Finger contact mode: sample center 32x32 patch
        const cx0 = Math.round(w * 0.5 - 16);
        const cx1 = Math.round(w * 0.5 + 16);
        const cy0 = Math.round(h * 0.5 - 16);
        const cy1 = Math.round(h * 0.5 + 16);

        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        for (let y = cy0; y < cy1; y++) {
          const rowOffset = y * w * 4;
          for (let x = cx0; x < cx1; x++) {
            const ptr = rowOffset + x * 4;
            sumR += data[ptr];
            sumG += data[ptr + 1];
            sumB += data[ptr + 2];
            count++;
          }
        }
        rMean = sumR / count;
        gMean = sumG / count;
        bMean = sumB / count;
      }

      // Feed into DSP resampler
      resamplerR.push(frame.timestampMs, rMean, (uTs, uR) => {
        resamplerG.push(uTs, gMean, (_tsG, uG) => {
          resamplerB.push(uTs, bMean, (_tsB, uB) => {
            let pulse = 0.0;
            if (target.mode === 'face') {
              pulse = pos.process(uR, uG, uB);
            } else {
              pulse = -(uG / (uR + 1e-4));
            }

            rawPpgBuffer.push(pulse);
            const filtered = filter.process(pulse);
            sqi.push(filtered);
            peakDetector.process(uTs, filtered);
          });
        });
      });
    });

    // Evaluate telemetry for session
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
        } else if (sqiEval.sqi >= 0.30) {
          estimatedBpm = fftBpm;
        } else {
          estimatedBpm = 0.50 * peakBpm + 0.50 * fftBpm;
        }
      }
    }

    const estimatedRpm = coordinator.estimateRespirationRate(rawPpgBuffer, 30.0);

    gtHrList.push(target.groundTruthHr);
    estHrList.push(estimatedBpm);
    gtRrList.push(target.groundTruthRr);
    estRrList.push(estimatedRpm);

    const hrErr = Math.abs(estimatedBpm - target.groundTruthHr);
    const rrErr = Math.abs(estimatedRpm - target.groundTruthRr);

    console.log(`[Session #${target.sessionId.toString().padStart(2)}] Mode: ${target.mode.padEnd(6)} | Fitz: ${target.fitzpatrick} | HR GT: ${target.groundTruthHr.toFixed(1).padStart(5)} BPM -> Est: ${estimatedBpm.toFixed(1).padStart(5)} BPM (Err: ${hrErr.toFixed(2)} BPM) | RR GT: ${target.groundTruthRr.toFixed(1).padStart(4)} -> Est: ${estimatedRpm.toFixed(1).padStart(4)} RPM (Err: ${rrErr.toFixed(2)} RPM)`);
  }

  // Statistical summary
  const calcPearson = (x: number[], y: number[]) => {
    const n = x.length;
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
    const mx = sx / n, my = sy / n;
    let num = 0, dx2 = 0, dy2 = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      num += dx * dy;
      dx2 += dx * dx;
      dy2 += dy * dy;
    }
    return num / Math.sqrt(dx2 * dy2);
  };

  let totalHrErr = 0;
  let totalRrErr = 0;
  for (let i = 0; i < gtHrList.length; i++) {
    totalHrErr += Math.abs(estHrList[i] - gtHrList[i]);
    totalRrErr += Math.abs(estRrList[i] - gtRrList[i]);
  }

  const hrPearson = calcPearson(gtHrList, estHrList);
  const rrPearson = calcPearson(gtRrList, estRrList);
  const hrMae = totalHrErr / gtHrList.length;
  const rrMae = totalRrErr / gtRrList.length;
  const tTotal = Date.now() - tStart;

  console.log(`\n---------------------------------------------------------------`);
  console.log(`SUMMARY ACROSS ${totalFramesProcessed.toLocaleString()} 2D VIDEO FRAMES:`);
  console.log(`  Heart Rate Pearson r:   ${hrPearson.toFixed(5)} (MAE: ${hrMae.toFixed(2)} BPM)`);
  console.log(`  Respiration Pearson r:  ${rrPearson.toFixed(5)} (MAE: ${rrMae.toFixed(2)} RPM)`);
  console.log(`  Execution Time:         ${(tTotal / 1000).toFixed(2)}s (${(tTotal / totalFramesProcessed).toFixed(2)} ms/frame)`);
  console.log(`===============================================================\n`);

  return {
    totalFrames: totalFramesProcessed,
    totalSessions: sessionTargets.length,
    heartRatePearsonR: hrPearson,
    heartRateMae: hrMae,
    respirationPearsonR: rrPearson,
    respirationMae: rrMae,
    durationMs: tTotal,
  };
}
