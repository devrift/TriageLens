import { describe, expect, it } from 'vitest';
import { StressTestingHarness } from '../StressTestingHarness';

describe('TriageLens Extreme Edge Case & Stress Testing Battery', () => {
  it('passes all physiological and environmental edge cases with clinical accuracy', () => {
    const results = StressTestingHarness.runStressBattery();

    console.log('\n===============================================================');
    console.log('       TRIAGELENS EXTREME PHYSIOLOGICAL STRESS BATTERY         ');
    console.log('===============================================================');

    for (const res of results) {
      const statusIcon = res.passed ? '✓ PASS' : '✗ FAIL';
      console.log(`[${statusIcon}] [${res.category.padEnd(20)}] ${res.testName.padEnd(38)} | Ground Truth: ${res.groundTruth.toString().padStart(5)} | Measured: ${res.measured.toString().padStart(5)} | Error: ${res.error.toFixed(2)}`);
      expect(res.passed).toBe(true);
    }

    console.log('===============================================================\n');
  });
});
