import { describe, it, expect } from 'vitest';
import { PhysiomeDatasetGenerator } from '../PhysiomeDatasetGenerator';
import { ClinicalBenchmarkHarness } from '../ClinicalBenchmarkHarness';

describe('Clinical Benchmark Harness Baseline', () => {
  it('evaluates baseline accuracy across diverse Fitzpatrick phototypes and physiological conditions', () => {
    const generator = new PhysiomeDatasetGenerator(12345);
    const numVectors = 100;
    const samples = [];

    for (let i = 0; i < numVectors; i++) {
      samples.push(generator.generateVector({}, i));
    }

    const report = ClinicalBenchmarkHarness.evaluateBatch(samples);

    console.log('=== CLINICAL BENCHMARK REPORT (BASELINE 100 VECTORS) ===');
    console.log(`Heart Rate Pearson r: ${report.heartRate.pearsonR.toFixed(4)}, MAE: ${report.heartRate.mae.toFixed(2)} BPM, RMSE: ${report.heartRate.rmse.toFixed(2)} BPM, Accuracy <= 3 BPM: ${report.heartRate.accuracyWithinTolerancePct.toFixed(1)}%`);
    console.log(`Respiration Pearson r: ${report.respirationRate.pearsonR.toFixed(4)}, MAE: ${report.respirationRate.mae.toFixed(2)} RPM, Accuracy <= 2 RPM: ${report.respirationRate.accuracyWithinTolerancePct.toFixed(1)}%`);
    console.log(`Facial Asymmetry Pearson r: ${report.strokeAsymmetry.pearsonR.toFixed(4)}, MAE: ${report.strokeAsymmetry.mae.toFixed(4)}`);
    console.log('Fitzpatrick Stratification:', JSON.stringify(report.fitzpatrickStratification, null, 2));

    expect(report.totalVectors).toBe(numVectors);
  });
});
