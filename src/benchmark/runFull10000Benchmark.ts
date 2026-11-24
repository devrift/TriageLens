/**
 * TriageLens 10,000 Vector Clinical Telemetry Benchmark Execution
 * 
 * Evaluates 10,000 multi-parameter synthetic physiological vectors spanning:
 * - Fitzpatrick phototypes I through VI
 * - Heart Rates 40 - 180 BPM (bradycardia to severe tachycardia)
 * - Respiration Rates 8 - 36 RPM (bradypnea to tachypnea)
 * - SNRs -5 dB to +25 dB
 * - Hardware timestamp jitter (20 - 45 ms) and 50Hz/60Hz AC lighting flicker
 * - 3D head poses with yaw, pitch, roll rotations up to +- 25 degrees
 * - Pathological bilateral facial paresis deltas
 */

import { PhysiomeDatasetGenerator } from './PhysiomeDatasetGenerator';
import { ClinicalBenchmarkHarness, BenchmarkReport } from './ClinicalBenchmarkHarness';

export function runBenchmark10000(batchSize: number = 10000, seed: number = 99887766): BenchmarkReport {
  console.log(`\n===============================================================`);
  console.log(`  TRIAGELENS 10,000 CLINICAL VALIDATION BENCHMARK EXECUTION   `);
  console.log(`  ISO 80601-2-61 & ANSI/AAMI EC13 Compliance Suite            `);
  console.log(`===============================================================\n`);
  console.log(`Generating and evaluating ${batchSize.toLocaleString()} vectors across Fitzpatrick I-VI...`);

  const generator = new PhysiomeDatasetGenerator(seed);
  const CHUNK_SIZE = 500;
  const numChunks = Math.ceil(batchSize / CHUNK_SIZE);

  const tStart = Date.now();

  for (let c = 0; c < numChunks; c++) {
    const currentChunkSize = Math.min(CHUNK_SIZE, batchSize - c * CHUNK_SIZE);
    const chunkSamples = [];
    for (let i = 0; i < currentChunkSize; i++) {
      const vectorId = c * CHUNK_SIZE + i;
      chunkSamples.push(generator.generateVector({}, vectorId));
    }

    ClinicalBenchmarkHarness.evaluateBatch(chunkSamples);

    if ((c + 1) % 4 === 0 || c === numChunks - 1) {
      const progressVectors = (c + 1) * CHUNK_SIZE;
      const progressPct = Math.min(100, (progressVectors / batchSize) * 100);
      console.log(`[Batch Progress: ${progressPct.toFixed(0)}%] Evaluated ${Math.min(progressVectors, batchSize).toLocaleString()} / ${batchSize.toLocaleString()} vectors...`);
    }
  }

  // Run the full combined evaluation
  // Generate unified evaluation sample set or run combined accumulator
  const fullSamples = [];
  generator.setSeed(seed);
  for (let i = 0; i < batchSize; i++) {
    fullSamples.push(generator.generateVector({}, i));
  }

  const finalReport = ClinicalBenchmarkHarness.evaluateBatch(fullSamples);
  const tTotalSec = ((Date.now() - tStart) / 1000).toFixed(2);

  console.log(`\n===============================================================`);
  console.log(`                 10,000 VECTOR BENCHMARK REPORT                `);
  console.log(`===============================================================`);
  console.log(`Total Vectors Evaluated: ${finalReport.totalVectors.toLocaleString()}`);
  console.log(`Benchmark Runtime:       ${tTotalSec}s (${(finalReport.durationMs / finalReport.totalVectors).toFixed(2)} ms/vector)`);
  console.log(`---------------------------------------------------------------`);
  console.log(`HEART RATE (BPM):`);
  console.log(`  Pearson Correlation (r):  ${finalReport.heartRate.pearsonR.toFixed(5)} (Target: >= 0.9800)`);
  console.log(`  Mean Absolute Error (MAE): ${finalReport.heartRate.mae.toFixed(2)} BPM (Target: <= 1.80 BPM)`);
  console.log(`  Root Mean Square Error:   ${finalReport.heartRate.rmse.toFixed(2)} BPM`);
  console.log(`  Accuracy within +-3 BPM:   ${finalReport.heartRate.accuracyWithinTolerancePct.toFixed(2)}% (Target: >= 98.0%)`);
  console.log(`---------------------------------------------------------------`);
  console.log(`RESPIRATION RATE (RPM):`);
  console.log(`  Pearson Correlation (r):  ${finalReport.respirationRate.pearsonR.toFixed(5)} (Target: >= 0.9800)`);
  console.log(`  Mean Absolute Error (MAE): ${finalReport.respirationRate.mae.toFixed(2)} RPM`);
  console.log(`  Accuracy within +-2 RPM:   ${finalReport.respirationRate.accuracyWithinTolerancePct.toFixed(2)}%`);
  console.log(`---------------------------------------------------------------`);
  console.log(`3D FACIAL STROKE ASYMMETRY (FAST PROTOCOL):`);
  console.log(`  Pearson Correlation (r):  ${finalReport.strokeAsymmetry.pearsonR.toFixed(5)} (Target: >= 0.9800)`);
  console.log(`  Mean Absolute Error (MAE): ${finalReport.strokeAsymmetry.mae.toFixed(5)}`);
  console.log(`  Accuracy within +-0.03:    ${finalReport.strokeAsymmetry.accuracyWithinTolerancePct.toFixed(2)}%`);
  console.log(`---------------------------------------------------------------`);
  console.log(`FITZPATRICK STRATIFICATION (HEART RATE):`);
  for (const [fitz, stats] of Object.entries(finalReport.fitzpatrickStratification)) {
    console.log(`  Type ${fitz.padEnd(3)}: N = ${stats.count.toString().padEnd(5)} | MAE = ${stats.hrMae.toFixed(2)} BPM | Pearson r = ${stats.hrPearsonR.toFixed(5)}`);
  }
  console.log(`---------------------------------------------------------------`);
  console.log(`SNR STRATIFICATION:`);
  console.log(`  Low SNR (< 6 dB):   N = ${finalReport.snrStratification.lowSnr.count.toString().padEnd(5)} | MAE = ${finalReport.snrStratification.lowSnr.hrMae.toFixed(2)} BPM`);
  console.log(`  Mid SNR (6-15 dB):  N = ${finalReport.snrStratification.midSnr.count.toString().padEnd(5)} | MAE = ${finalReport.snrStratification.midSnr.hrMae.toFixed(2)} BPM`);
  console.log(`  High SNR (> 15 dB): N = ${finalReport.snrStratification.highSnr.count.toString().padEnd(5)} | MAE = ${finalReport.snrStratification.highSnr.hrMae.toFixed(2)} BPM`);
  console.log(`===============================================================\n`);

  return finalReport;
}

if (typeof process !== 'undefined' && process.argv && process.argv[1] && process.argv[1].includes('runFull10000Benchmark')) {
  runBenchmark10000(10000);
}
