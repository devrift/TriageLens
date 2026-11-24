import { describe, it, expect } from 'vitest';
import { run10000VideoFrameEvaluation } from '../run10000FrameVideoEvaluation';

describe('10,000 Raw Video Frame Evaluation', () => {
  it('processes 10,000 raw 2D video frames (320x240 RGBA) through vision ROI & DSP with r >= 0.98', () => {
    const res = run10000VideoFrameEvaluation(44556677);

    expect(res.totalFrames).toBeGreaterThanOrEqual(10000);
    expect(res.heartRatePearsonR).toBeGreaterThanOrEqual(0.98);
    expect(res.heartRateMae).toBeLessThanOrEqual(1.80);
    expect(res.respirationPearsonR).toBeGreaterThanOrEqual(0.98);
    expect(res.respirationMae).toBeLessThanOrEqual(1.50);
  }, 90000); // 90 second timeout for 10,000 raw 2D pixel image frames
});
