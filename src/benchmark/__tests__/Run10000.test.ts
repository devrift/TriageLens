import { describe, it, expect } from 'vitest';
import { runBenchmark10000 } from '../runFull10000Benchmark';

describe('10,000 Vector Full Clinical Benchmark Suite', () => {
  it('executes 10,000 vector benchmark and achieves clinical accuracy invariants (r >= 0.98, MAE <= 1.8 BPM)', () => {
    // Run full 10,000 vector evaluation
    const report = runBenchmark10000(10000, 77889900);

    expect(report.totalVectors).toBe(10000);

    // Heart Rate Invariants
    expect(report.heartRate.pearsonR).toBeGreaterThanOrEqual(0.98);
    expect(report.heartRate.mae).toBeLessThanOrEqual(1.80);

    // Respiration Rate Invariants
    expect(report.respirationRate.pearsonR).toBeGreaterThanOrEqual(0.98);
    expect(report.respirationRate.mae).toBeLessThanOrEqual(1.50);

    // 3D Facial Asymmetry Invariants
    expect(report.strokeAsymmetry.pearsonR).toBeGreaterThanOrEqual(0.98);
    expect(report.strokeAsymmetry.mae).toBeLessThanOrEqual(0.01);
  }, 120000); // 2 minute timeout for 10,000 vectors
});
