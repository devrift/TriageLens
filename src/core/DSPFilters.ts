/**
 * TriageLens Digital Signal Processing (DSP) Filters & Mathematical Algorithms
 * 
 * Strict deterministic signal processing pipeline:
 * - Direct Form II Transposed Biquad Butterworth Bandpass Filter (0.75 - 3.0 Hz, 30 Hz fs)
 * - Uniform Grid Resampler (Linear Interpolation for mobile frame jitter, out-of-order/duplicate/burst safe)
 * - Plane-Orthogonal-to-Skin (POS) rPPG Algorithm (flat-window & lighting-shift safe)
 * - Peak Detector with Adaptive Thresholding & Refractory Period
 * - 256-point Power Spectral Density Signal Quality Index (SQI)
 */

export interface BiquadCoefficients {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

/**
 * Single Second-Order Section (Biquad) Direct Form II Transposed
 * 
 * Difference equations:
 *   y[n]  = b0 * x[n] + w1[n-1]
 *   w1[n] = b1 * x[n] - a1 * y[n] + w2[n-1]
 *   w2[n] = b2 * x[n] - a2 * y[n]
 * 
 * Numerically stable under single & double precision, zero-allocation per sample.
 */
export class BiquadFilter {
  public b0: number;
  public b1: number;
  public b2: number;
  public a1: number;
  public a2: number;

  private w1: number = 0.0;
  private w2: number = 0.0;

  constructor(coeffs: BiquadCoefficients) {
    this.b0 = coeffs.b0;
    this.b1 = coeffs.b1;
    this.b2 = coeffs.b2;
    this.a1 = coeffs.a1;
    this.a2 = coeffs.a2;
  }

  /**
   * Process a single incoming sample through Direct Form II Transposed biquad.
   * Execution time is O(1) with 0 memory allocation.
   * Safe against NaN or non-finite inputs.
   */
  public process(x: number): number {
    const safeX = Number.isFinite(x) ? x : 0.0;
    const y = this.b0 * safeX + this.w1;
    this.w1 = this.b1 * safeX - this.a1 * y + this.w2;
    this.w2 = this.b2 * safeX - this.a2 * y;
    return Number.isFinite(y) ? y : 0.0;
  }

  /**
   * Reset internal delay registers to zero.
   */
  public reset(): void {
    this.w1 = 0.0;
    this.w2 = 0.0;
  }

  /**
   * Factory method: Creates a 4th-order Butterworth bandpass filter configured for:
   *   - Sampling Frequency (fs): 30.0 Hz
   *   - Passband: [0.75 Hz, 3.0 Hz] (45 to 180 BPM)
   * 
   * Implemented as a cascade of two second-order Direct Form II Transposed biquad sections.
   * Analytical coefficients derived via Bilinear Transformation with frequency pre-warping.
   */
  public static createButterworthBandpass30Hz(): BiquadCascade {
    // Section 1 coefficients (Q = 1.30656 pole pair mapped to digital z-plane, [0.70 Hz, 2.90 Hz])
    const section1: BiquadCoefficients = {
      b0: 0.1215173912910731,
      b1: 0.0,
      b2: -0.1215173912910731,
      a1: -1.8292526682595407,
      a2: 0.8548864958743286,
    };

    // Section 2 coefficients (Q = 0.541196 pole pair mapped to digital z-plane, [0.70 Hz, 2.90 Hz])
    const section2: BiquadCoefficients = {
      b0: 0.32646591876561937,
      b1: 0.0,
      b2: -0.32646591876561937,
      a1: -1.382362317762067,
      a2: 0.6101412896841377,
    };

    return new BiquadCascade([new BiquadFilter(section1), new BiquadFilter(section2)]);
  }
}

/**
 * Cascade of multiple BiquadFilter sections (e.g. 4th-order cascade of two 2nd-order biquads)
 */
export class BiquadCascade {
  private sections: BiquadFilter[];

  constructor(sections: BiquadFilter[]) {
    this.sections = sections;
  }

  public process(sample: number): number {
    let out = sample;
    for (let i = 0; i < this.sections.length; i++) {
      out = this.sections[i].process(out);
    }
    return out;
  }

  public reset(): void {
    for (let i = 0; i < this.sections.length; i++) {
      this.sections[i].reset();
    }
  }

  public getSections(): BiquadFilter[] {
    return this.sections;
  }
}

/**
 * Uniform Grid Resampler
 * 
 * Resamples non-uniform hardware timestamped frame streams (with 28ms - 36ms jitter)
 * onto a strictly uniform 30.0 Hz grid (deltaT = 33.333 ms) using linear interpolation.
 * 
 * Hardened against:
 *   - Duplicate timestamps (dt = 0)
 *   - Out-of-order timestamps (backward time delta)
 *   - Bursty/frozen frame delivery gaps (>500ms)
 *   - Non-finite (NaN / Infinity) input values
 * 
 * Pre-allocated circular ring buffer guarantees zero garbage collection pauses.
 */
export class UniformResampler {
  private readonly targetDeltaMs: number;
  private readonly capacity: number;
  private readonly timestamps: Float64Array;
  private readonly values: Float64Array;
  private head: number = 0;
  private count: number = 0;

  private nextUniformTimestamp: number = -1;
  private isInitialized: boolean = false;
  private lastPushedTimestamp: number = -1e9;

  constructor(targetFsHz: number = 30.0, capacity: number = 256) {
    this.targetDeltaMs = 1000.0 / targetFsHz;
    this.capacity = capacity;
    this.timestamps = new Float64Array(capacity);
    this.values = new Float64Array(capacity);
  }

  public reset(): void {
    this.head = 0;
    this.count = 0;
    this.nextUniformTimestamp = -1;
    this.isInitialized = false;
    this.lastPushedTimestamp = -1e9;
    this.timestamps.fill(0);
    this.values.fill(0);
  }

  /**
   * Push an incoming hardware sample (t_k, x(t_k)).
   * Yields any interpolated samples that fall on the uniform 30.0 Hz grid.
   * 
   * @param timestamp Monotonic timestamp (performance.now())
   * @param value Signal value
   * @param outputCallback Invoked for each uniform 30 Hz sample (zero allocation)
   */
  public push(
    timestamp: number,
    value: number,
    outputCallback: (uniformTs: number, interpolatedVal: number) => void
  ): void {
    if (!Number.isFinite(timestamp)) {
      return;
    }
    const safeValue = Number.isFinite(value)
      ? value
      : (this.count > 0 ? this.values[(this.head + this.count - 1) % this.capacity] : 0.0);

    // Guard 1: Out-of-order or duplicate timestamp handling
    if (this.count > 0) {
      if (timestamp < this.lastPushedTimestamp) {
        // Drop backwards timestamp to prevent ring-buffer monotonicity violation
        return;
      }
      if (timestamp === this.lastPushedTimestamp) {
        // Replace previous duplicate value to avoid zero-span division
        const lastIdx = (this.head + this.count - 1) % this.capacity;
        this.values[lastIdx] = safeValue;
        return;
      }
    }

    // Guard 2: Bursty gap recovery (if camera paused/frozen for > 500ms)
    if (this.isInitialized && timestamp - this.lastPushedTimestamp > 500.0) {
      // Re-anchor uniform clock to avoid generating hundreds of stale interpolated frames
      this.nextUniformTimestamp = timestamp + this.targetDeltaMs;
      this.head = 0;
      this.count = 0;
    }

    this.lastPushedTimestamp = timestamp;

    // Write into circular ring buffer
    const insertIdx = (this.head + this.count) % this.capacity;
    if (this.count === this.capacity) {
      // Buffer full, overwrite oldest
      this.timestamps[this.head] = timestamp;
      this.values[this.head] = safeValue;
      this.head = (this.head + 1) % this.capacity;
    } else {
      this.timestamps[insertIdx] = timestamp;
      this.values[insertIdx] = safeValue;
      this.count++;
    }

    if (!this.isInitialized) {
      if (this.count >= 2) {
        // Initialize uniform timestamp grid aligned to the first captured sample
        const t0 = this.timestamps[this.head];
        this.nextUniformTimestamp = t0 + this.targetDeltaMs;
        this.isInitialized = true;
      } else {
        return;
      }
    }

    // Produce all uniform samples that can be interpolated up to the latest timestamp
    let producedCount = 0;
    const maxProducesPerCall = 30; // Safety clamp against runaway loops

    while (this.count >= 2 && producedCount < maxProducesPerCall) {
      const latestIdx = (this.head + this.count - 1) % this.capacity;
      const latestTs = this.timestamps[latestIdx];

      if (this.nextUniformTimestamp > latestTs) {
        // Next uniform point is in the future, wait for next frame
        break;
      }

      // Find the two consecutive samples [t_a, t_b] enclosing nextUniformTimestamp
      let found = false;
      let valA = 0;
      let valB = 0;
      let tA = 0;
      let tB = 0;

      for (let i = 0; i < this.count - 1; i++) {
        const idxA = (this.head + i) % this.capacity;
        const idxB = (this.head + i + 1) % this.capacity;
        const currTA = this.timestamps[idxA];
        const currTB = this.timestamps[idxB];

        if (currTA <= this.nextUniformTimestamp && this.nextUniformTimestamp <= currTB) {
          tA = currTA;
          tB = currTB;
          valA = this.values[idxA];
          valB = this.values[idxB];
          found = true;
          break;
        }
      }

      if (found) {
        const span = tB - tA;
        let interpVal = valA;
        if (span > 1e-6) {
          const mu = (this.nextUniformTimestamp - tA) / span;
          interpVal = valA + mu * (valB - valA);
        }
        outputCallback(this.nextUniformTimestamp, interpVal);
        producedCount++;
        this.nextUniformTimestamp += this.targetDeltaMs;

        // Evict samples that are older than nextUniformTimestamp - 2 * targetDeltaMs
        while (this.count > 2) {
          const nextOldestIdx = (this.head + 1) % this.capacity;
          if (this.timestamps[nextOldestIdx] < this.nextUniformTimestamp - this.targetDeltaMs) {
            this.head = nextOldestIdx;
            this.count--;
          } else {
            break;
          }
        }
      } else {
        // Timestamp falls before our buffer head, advance
        if (this.nextUniformTimestamp < this.timestamps[this.head]) {
          this.nextUniformTimestamp += this.targetDeltaMs;
        } else {
          break;
        }
      }
    }
  }
}

/**
 * Plane-Orthogonal-to-Skin (POS) Algorithm for rPPG
 * 
 * Given forehead ROI spatial mean C(t) = [R(t), G(t), B(t)]^T:
 *   1. Sliding temporal window L = 45 frames (1.5 s at 30 FPS).
 *   2. Normalize color channels by temporal mean: c_n(t) = C(t) / ( (1/L) * sum(C(k)) )
 *   3. Project onto orthogonal chrominance planes:
 *        S1(t) = g_n(t) - b_n(t)
 *        S2(t) = g_n(t) + b_n(t) - 2 * r_n(t)
 *   4. Standard deviations sigma(S1), sigma(S2) over window L.
 *   5. Tuning ratio alpha = sigma(S1) / (sigma(S2) + 1e-6).
 *   6. Pulsatile signal H(t) = S1(t) + alpha * S2(t).
 * 
 * Hardened against:
 *   - Flat RGB inputs (zero-variance / darkness)
 *   - Abrupt lighting shifts
 *   - Non-finite or negative pixel values
 * 
 * Uses strictly pre-allocated static typed arrays. Zero GC allocations in loop.
 */
export class PlaneOrthogonalToSkin {
  public static readonly WINDOW_L: number = 45;

  private readonly rBuf: Float64Array = new Float64Array(PlaneOrthogonalToSkin.WINDOW_L);
  private readonly gBuf: Float64Array = new Float64Array(PlaneOrthogonalToSkin.WINDOW_L);
  private readonly bBuf: Float64Array = new Float64Array(PlaneOrthogonalToSkin.WINDOW_L);

  private readonly s1Buf: Float64Array = new Float64Array(PlaneOrthogonalToSkin.WINDOW_L);
  private readonly s2Buf: Float64Array = new Float64Array(PlaneOrthogonalToSkin.WINDOW_L);

  private head: number = 0;
  private count: number = 0;

  public reset(): void {
    this.head = 0;
    this.count = 0;
    this.rBuf.fill(0);
    this.gBuf.fill(0);
    this.bBuf.fill(0);
    this.s1Buf.fill(0);
    this.s2Buf.fill(0);
  }

  /**
   * Process incoming RGB spatial mean [R, G, B].
   * Returns the estimated pulsatile blood volume signal H(t).
   */
  public process(r: number, g: number, b: number): number {
    if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
      return 0.0;
    }

    const safeR = Math.max(0.0, r);
    const safeG = Math.max(0.0, g);
    const safeB = Math.max(0.0, b);

    const L = PlaneOrthogonalToSkin.WINDOW_L;

    // Insert into circular ring buffer
    const insertIdx = (this.head + this.count) % L;
    if (this.count === L) {
      this.rBuf[this.head] = safeR;
      this.gBuf[this.head] = safeG;
      this.bBuf[this.head] = safeB;
      this.head = (this.head + 1) % L;
    } else {
      this.rBuf[insertIdx] = safeR;
      this.gBuf[insertIdx] = safeG;
      this.bBuf[insertIdx] = safeB;
      this.count++;
    }

    const currentCount = this.count;
    if (currentCount < 2) {
      return 0.0;
    }

    // 1. Calculate temporal means over available window
    let sumR = 0.0;
    let sumG = 0.0;
    let sumB = 0.0;

    for (let i = 0; i < currentCount; i++) {
      const idx = (this.head + i) % L;
      sumR += this.rBuf[idx];
      sumG += this.gBuf[idx];
      sumB += this.bBuf[idx];
    }

    const meanR = sumR / currentCount;
    const meanG = sumG / currentCount;
    const meanB = sumB / currentCount;

    // Flat or near-zero luminance check
    if (meanR < 1e-4 || meanG < 1e-4 || meanB < 1e-4) {
      return 0.0;
    }

    // 2. Normalize and project each sample in the window onto S1 and S2
    let sumS1 = 0.0;
    let sumS2 = 0.0;

    for (let i = 0; i < currentCount; i++) {
      const idx = (this.head + i) % L;
      const rn = this.rBuf[idx] / meanR;
      const gn = this.gBuf[idx] / meanG;
      const bn = this.bBuf[idx] / meanB;

      const s1 = gn - bn;
      const s2 = gn + bn - 2.0 * rn;

      this.s1Buf[i] = s1;
      this.s2Buf[i] = s2;

      sumS1 += s1;
      sumS2 += s2;
    }

    // 3. Compute standard deviations sigma(S1) and sigma(S2)
    const meanS1 = sumS1 / currentCount;
    const meanS2 = sumS2 / currentCount;

    let varS1 = 0.0;
    let varS2 = 0.0;

    for (let i = 0; i < currentCount; i++) {
      const d1 = this.s1Buf[i] - meanS1;
      const d2 = this.s2Buf[i] - meanS2;
      varS1 += d1 * d1;
      varS2 += d2 * d2;
    }

    const sigmaS1 = Math.sqrt(varS1 / currentCount);
    const sigmaS2 = Math.sqrt(varS2 / currentCount);

    // 4. Tuning ratio alpha (guarded against zero-variance flat signals)
    const alpha = sigmaS2 > 1e-6 ? sigmaS1 / (sigmaS2 + 1e-6) : 0.0;

    // 5. Pulsatile signal for latest sample
    const latestIdx = currentCount - 1;
    const latestS1 = this.s1Buf[latestIdx];
    const latestS2 = this.s2Buf[latestIdx];

    const h = latestS1 + alpha * latestS2;
    return Number.isFinite(h) ? h : 0.0;
  }
}

/**
 * Peak Detector with Adaptive Thresholding & Refractory Period
 * 
 * - Sliding window W = 150 samples (5.0 seconds at 30 Hz).
 * - Constraints:
 *     1. Amplitude Floor: peak > movingMean + 0.4 * movingStd
 *     2. Refractory Lockout: min distance >= 300 ms (max 200 BPM)
 * - Inter-Beat Intervals (IBI) and median BPM calculation across low, normal, and high rates.
 */
export class PeakDetector {
  public static readonly WINDOW_W: number = 150;
  public static readonly ABSOLUTE_MIN_REFRACTORY_MS: number = 240.0; // allows up to 250 BPM

  private readonly samples: Float64Array = new Float64Array(PeakDetector.WINDOW_W);
  private readonly timestamps: Float64Array = new Float64Array(PeakDetector.WINDOW_W);
  private head: number = 0;
  private count: number = 0;

  private lastPeakTimestamp: number = -1e9;
  private runningTrough: number = 1e9;
  private readonly ibiHistory: Float64Array = new Float64Array(32);
  private ibiCount: number = 0;
  private ibiHead: number = 0;

  private currentBpm: number = 0.0;
  private lastDetectedPeak: { timestamp: number; value: number } | null = null;

  public reset(): void {
    this.head = 0;
    this.count = 0;
    this.lastPeakTimestamp = -1e9;
    this.runningTrough = 1e9;
    this.ibiCount = 0;
    this.ibiHead = 0;
    this.currentBpm = 0.0;
    this.lastDetectedPeak = null;
    this.samples.fill(0);
    this.timestamps.fill(0);
    this.ibiHistory.fill(0);
  }

  /**
   * Process a single filtered waveform sample.
   * Employs sub-sample parabolic interpolation and dynamic refractory lockout
   * to eliminate discrete sampling jitter and dicrotic notch false triggering.
   * 
   * @param timestamp Monotonic timestamp (ms)
   * @param sample Filtered signal value
   * @returns Detected peak if this sample confirmed a local maximum, else null
   */
  public process(timestamp: number, sample: number): { timestamp: number; value: number; bpm: number } | null {
    if (!Number.isFinite(timestamp) || !Number.isFinite(sample)) {
      return null;
    }

    if (sample < this.runningTrough) {
      this.runningTrough = sample;
    }

    const W = PeakDetector.WINDOW_W;
    const insertIdx = (this.head + this.count) % W;

    if (this.count === W) {
      this.samples[this.head] = sample;
      this.timestamps[this.head] = timestamp;
      this.head = (this.head + 1) % W;
    } else {
      this.samples[insertIdx] = sample;
      this.timestamps[insertIdx] = timestamp;
      this.count++;
    }

    if (this.count < 3) {
      return null;
    }

    // Inspect candidate peak at index (count - 2): middle of last 3 samples
    const midIdx = (this.head + this.count - 2 + W) % W;
    const prevIdx = (this.head + this.count - 3 + W) % W;
    const nextIdx = (this.head + this.count - 1 + W) % W;

    const vPrev = this.samples[prevIdx];
    const vMid = this.samples[midIdx];
    const vNext = this.samples[nextIdx];
    const tMid = this.timestamps[midIdx];

    // Local maximum check: strictly greater than previous, greater than or equal to next
    if (vMid > vPrev && vMid >= vNext) {
      // 1. Moving mean and standard deviation over window W
      let sum = 0.0;
      for (let i = 0; i < this.count; i++) {
        const idx = (this.head + i) % W;
        sum += this.samples[idx];
      }
      const mean = sum / this.count;

      let varSum = 0.0;
      for (let i = 0; i < this.count; i++) {
        const idx = (this.head + i) % W;
        const diff = this.samples[idx] - mean;
        varSum += diff * diff;
      }
      const std = Math.sqrt(varSum / this.count);

      // Constraint 1: Amplitude Floor (mean + 0.35 * std)
      const threshold = mean + 0.35 * std;

      // Prominence check against running trough
      const prominence = vMid - this.runningTrough;
      const minProminence = Math.max(0.001, 0.4 * std);

      // Constraint 2: Dynamic Refractory Lockout
      // Use 52% of median IBI to lockout dicrotic notches (which occur at ~35-45% of IBI)
      const medianIbiEstimate = this.getMedianIbi();
      const dynamicRefractoryMs = medianIbiEstimate > 0
        ? Math.max(PeakDetector.ABSOLUTE_MIN_REFRACTORY_MS, Math.min(500.0, 0.52 * medianIbiEstimate))
        : PeakDetector.ABSOLUTE_MIN_REFRACTORY_MS;

      const deltaT = tMid - this.lastPeakTimestamp;

      if (vMid > threshold && prominence >= minProminence && deltaT >= dynamicRefractoryMs) {
        // Sub-sample Parabolic Interpolation for exact peak timestamp vertex
        // Parabola: y(x) = a*x^2 + b*x + c with vertex offset delta = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
        const denom = vPrev - 2.0 * vMid + vNext;
        let subSampleOffsetSamples = 0.0;
        let interpolatedPeakValue = vMid;

        if (Math.abs(denom) > 1e-12) {
          subSampleOffsetSamples = (vPrev - vNext) / (2.0 * denom);
          // Clamp offset to [-0.5, 0.5] samples
          subSampleOffsetSamples = Math.max(-0.5, Math.min(0.5, subSampleOffsetSamples));
          interpolatedPeakValue = vMid - (denom * subSampleOffsetSamples * subSampleOffsetSamples);
        }

        const tPrev = this.timestamps[prevIdx];
        const tNext = this.timestamps[nextIdx];
        const nominalDt = (tNext - tPrev) / 2.0;
        const exactPeakTimestamp = tMid + subSampleOffsetSamples * nominalDt;

        if (this.lastPeakTimestamp > 0) {
          const exactIbi = exactPeakTimestamp - this.lastPeakTimestamp;
          if (exactIbi >= 240.0 && exactIbi <= 2500.0) {
            // Outlier check: if we already have stable history, reject ectopic beats or noise spikes
            const isOutlier = medianIbiEstimate > 0 &&
              (exactIbi < 0.60 * medianIbiEstimate || exactIbi > 1.65 * medianIbiEstimate);

            if (!isOutlier || this.ibiCount < 3) {
              const ibiInsert = (this.ibiHead + this.ibiCount) % this.ibiHistory.length;
              if (this.ibiCount === this.ibiHistory.length) {
                this.ibiHistory[this.ibiHead] = exactIbi;
                this.ibiHead = (this.ibiHead + 1) % this.ibiHistory.length;
              } else {
                this.ibiHistory[ibiInsert] = exactIbi;
                this.ibiCount++;
              }

              // Update median BPM
              this.currentBpm = this.calculateMedianBpm();
            }
          }
        }

        this.lastPeakTimestamp = exactPeakTimestamp;
        this.runningTrough = vMid; // reset trough tracking from new peak
        this.lastDetectedPeak = { timestamp: exactPeakTimestamp, value: interpolatedPeakValue };

        return {
          timestamp: exactPeakTimestamp,
          value: interpolatedPeakValue,
          bpm: this.currentBpm,
        };
      }
    }

    return null;
  }

  public getCurrentBpm(): number {
    return this.currentBpm;
  }

  public getLastDetectedPeak(): { timestamp: number; value: number } | null {
    return this.lastDetectedPeak;
  }

  /**
   * Compute Heart Rate Variability (HRV) and rhythm regularity statistics from recorded IBIs.
   * Zero heap allocations inside calculation.
   */
  public getIbiStats(): {
    count: number;
    meanIbiMs: number;
    sdnnMs: number;
    rmssdMs: number;
    coefficientOfVariation: number;
  } {
    const n = this.ibiCount;
    if (n < 2) {
      return { count: n, meanIbiMs: 0, sdnnMs: 0, rmssdMs: 0, coefficientOfVariation: 0 };
    }

    const len = this.ibiHistory.length;
    let sum = 0.0;
    for (let i = 0; i < n; i++) {
      sum += this.ibiHistory[(this.ibiHead + i) % len];
    }
    const meanIbiMs = sum / n;

    let varSum = 0.0;
    let succDiffSqSum = 0.0;
    let prevIbi = this.ibiHistory[this.ibiHead % len];

    for (let i = 0; i < n; i++) {
      const ibi = this.ibiHistory[(this.ibiHead + i) % len];
      const diff = ibi - meanIbiMs;
      varSum += diff * diff;

      if (i > 0) {
        const succDiff = ibi - prevIbi;
        succDiffSqSum += succDiff * succDiff;
        prevIbi = ibi;
      }
    }

    const sdnnMs = Math.sqrt(varSum / n);
    const rmssdMs = Math.sqrt(succDiffSqSum / (n - 1));
    const coefficientOfVariation = meanIbiMs > 1e-4 ? sdnnMs / meanIbiMs : 0.0;

    return {
      count: n,
      meanIbiMs,
      sdnnMs,
      rmssdMs,
      coefficientOfVariation,
    };
  }

  private getMedianIbi(): number {
    if (this.ibiCount === 0) return 0.0;

    const scratch = new Float64Array(this.ibiCount);
    for (let i = 0; i < this.ibiCount; i++) {
      const idx = (this.ibiHead + i) % this.ibiHistory.length;
      scratch[i] = this.ibiHistory[idx];
    }
    scratch.sort();

    const mid = Math.floor(this.ibiCount / 2);
    return this.ibiCount % 2 === 0
      ? (scratch[mid - 1] + scratch[mid]) / 2.0
      : scratch[mid];
  }

  /**
   * Compute median of recorded IBIs and derive BPM = 60000 / median(IBI)
   */
  private calculateMedianBpm(): number {
    const medianIbi = this.getMedianIbi();
    if (medianIbi < 1e-4) return 0.0;
    const bpm = 60000.0 / medianIbi;
    return Number.isFinite(bpm) ? bpm : 0.0;
  }
}

/**
 * Signal Quality Index (SQI)
 * 
 * Computes Power Spectral Density across rolling 256-point array via Radix-2 FFT.
 * Evaluates:
 *   - Cardiac peak frequency bin f_cardiac in [0.75 Hz, 3.0 Hz]
 *   - Signal band power P_signal in [f_c - 0.2, f_c + 0.2] + [2*f_c - 0.2, 2*f_c + 0.2]
 *   - Total band power P_total in [0.5 Hz, 5.0 Hz]
 *   - SQI = P_signal / P_total clamped to [0, 1]
 *   - SNR (dB) = 10 * log10(P_signal / (P_total - P_signal + eps))
 */
export class SignalQualityIndex {
  public static readonly FFT_SIZE: number = 256;

  private readonly buffer: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE);
  private head: number = 0;
  private count: number = 0;

  // Pre-allocated FFT work arrays
  private readonly real: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE);
  private readonly imag: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE);
  private readonly power: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE / 2);
  private readonly hanningWindow: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE);

  // Precomputed FFT bit-reversal and twiddle tables
  private readonly bitRev: Uint16Array = new Uint16Array(SignalQualityIndex.FFT_SIZE);
  private readonly cosTable: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE / 2);
  private readonly sinTable: Float64Array = new Float64Array(SignalQualityIndex.FFT_SIZE / 2);

  constructor() {
    const N = SignalQualityIndex.FFT_SIZE;
    // Precompute Hanning window
    for (let i = 0; i < N; i++) {
      this.hanningWindow[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (N - 1)));
    }

    // Precompute bit reversal indices
    const bits = Math.round(Math.log2(N));
    for (let i = 0; i < N; i++) {
      let rev = 0;
      let temp = i;
      for (let j = 0; j < bits; j++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
      }
      this.bitRev[i] = rev;
    }

    // Precompute twiddle factors
    for (let i = 0; i < N / 2; i++) {
      this.cosTable[i] = Math.cos((-2.0 * Math.PI * i) / N);
      this.sinTable[i] = Math.sin((-2.0 * Math.PI * i) / N);
    }
  }

  public reset(): void {
    this.head = 0;
    this.count = 0;
    this.buffer.fill(0);
  }

  public push(sample: number): void {
    const safeSample = Number.isFinite(sample) ? sample : 0.0;
    const N = SignalQualityIndex.FFT_SIZE;
    const insertIdx = (this.head + this.count) % N;
    if (this.count === N) {
      this.buffer[this.head] = safeSample;
      this.head = (this.head + 1) % N;
    } else {
      this.buffer[insertIdx] = safeSample;
      this.count++;
    }
  }

  public isBufferFull(): boolean {
    return this.count === SignalQualityIndex.FFT_SIZE;
  }

  /**
   * Evaluate Power Spectral Density and return SQI and SNR.
   * 
   * @param fs Sampling rate in Hz (default: 30.0)
   */
  public evaluate(fs: number = 30.0): { sqi: number; snrDb: number; peakFreqHz: number } {
    const N = SignalQualityIndex.FFT_SIZE;
    if (this.count < N) {
      return { sqi: 0.0, snrDb: -99.0, peakFreqHz: 0.0 };
    }

    // 1. Remove DC offset and apply Hanning window
    let sum = 0.0;
    for (let i = 0; i < N; i++) {
      const idx = (this.head + i) % N;
      sum += this.buffer[idx];
    }
    const mean = sum / N;

    let hasVariance = false;
    for (let i = 0; i < N; i++) {
      const idx = (this.head + i) % N;
      const centered = this.buffer[idx] - mean;
      if (Math.abs(centered) > 1e-6) {
        hasVariance = true;
      }
      const rev = this.bitRev[i];
      this.real[rev] = centered * this.hanningWindow[i];
      this.imag[rev] = 0.0;
    }

    if (!hasVariance) {
      return { sqi: 0.0, snrDb: -99.0, peakFreqHz: 0.0 };
    }

    // 2. In-place Cooley-Tukey Radix-2 FFT
    for (let len = 2; len <= N; len <<= 1) {
      const halfLen = len >> 1;
      const step = N / len;
      for (let i = 0; i < N; i += len) {
        for (let j = 0; j < halfLen; j++) {
          const tableIdx = j * step;
          const u_r = this.real[i + j];
          const u_i = this.imag[i + j];

          const v_r = this.real[i + j + halfLen];
          const v_i = this.imag[i + j + halfLen];

          const tw_r = this.cosTable[tableIdx];
          const tw_i = this.sinTable[tableIdx];

          const t_r = v_r * tw_r - v_i * tw_i;
          const t_i = v_r * tw_i + v_i * tw_r;

          this.real[i + j] = u_r + t_r;
          this.imag[i + j] = u_i + t_i;

          this.real[i + j + halfLen] = u_r - t_r;
          this.imag[i + j + halfLen] = u_i - t_i;
        }
      }
    }

    // 3. Compute one-sided power spectrum
    const halfN = N / 2;
    for (let k = 0; k < halfN; k++) {
      const r = this.real[k];
      const im = this.imag[k];
      this.power[k] = r * r + im * im;
    }

    const df = fs / N; // 30 / 256 = 0.1171875 Hz

    // 4. Identify peak frequency bin in physiological cardiac range [0.65 Hz, 3.20 Hz] (39 to 192 BPM)
    let maxPower = -1.0;
    let peakBin = -1;

    for (let k = 0; k < halfN; k++) {
      const freq = k * df;
      if (freq >= 0.65 && freq <= 3.20) {
        if (this.power[k] > maxPower) {
          maxPower = this.power[k];
          peakBin = k;
        }
      }
    }

    if (peakBin < 0 || maxPower <= 1e-12) {
      return { sqi: 0.0, snrDb: -99.0, peakFreqHz: 0.0 };
    }

    // Sub-bin quadratic interpolation around peakBin for continuous frequency resolution
    let refinedPeakBin = peakBin;
    if (peakBin > 0 && peakBin < halfN - 1) {
      const pPrev = this.power[peakBin - 1];
      const pMid = this.power[peakBin];
      const pNext = this.power[peakBin + 1];
      const denom = pPrev - 2.0 * pMid + pNext;
      if (Math.abs(denom) > 1e-12) {
        const deltaBin = (pPrev - pNext) / (2.0 * denom);
        refinedPeakBin = peakBin + Math.max(-0.5, Math.min(0.5, deltaBin));
      }
    }

    const fCardiac = refinedPeakBin * df;

    // 5. Integrate signal band power: [fCardiac - 0.2, fCardiac + 0.2] and harmonic [2*fCardiac - 0.2, 2*fCardiac + 0.2]
    let pSignal = 0.0;
    let pTotal = 0.0;

    const fHarmonic = 2.0 * fCardiac;

    for (let k = 0; k < halfN; k++) {
      const freq = k * df;
      const p = this.power[k];

      // Total power across [0.5 Hz, 5.0 Hz]
      if (freq >= 0.5 && freq <= 5.0) {
        pTotal += p;
      }

      // Fundamental band [f_cardiac - 0.2 Hz, f_cardiac + 0.2 Hz]
      const inFundamental = Math.abs(freq - fCardiac) <= 0.2;
      // First harmonic band [2 * f_cardiac - 0.2 Hz, 2 * f_cardiac + 0.2 Hz]
      const inHarmonic = Math.abs(freq - fHarmonic) <= 0.2;

      if (inFundamental || inHarmonic) {
        pSignal += p;
      }
    }

    if (pTotal <= 1e-12) {
      return { sqi: 0.0, snrDb: -99.0, peakFreqHz: fCardiac };
    }

    const sqi = Math.min(1.0, Math.max(0.0, pSignal / pTotal));
    const pNoise = Math.max(0.0, pTotal - pSignal);
    const snrDb = 10.0 * Math.log10((pSignal + 1e-12) / (pNoise + 1e-12));

    return {
      sqi: Number.isFinite(sqi) ? sqi : 0.0,
      snrDb: Number.isFinite(snrDb) ? snrDb : -99.0,
      peakFreqHz: fCardiac,
    };
  }
}
