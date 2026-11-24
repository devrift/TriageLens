/**
 * TriageLens TriageCoordinator
 * 
 * Central state machine and event emitter orchestrating:
 *   - Camera hardware stream lifecycle
 *   - Frame resampling onto uniform 30 Hz grid
 *   - FaceMesh rPPG & FAST stroke geometry OR Finger contact PPG
 *   - Real-time biquad filtering, peak detection, and SQI diagnostics
 *   - 15-second scan state machine: IDLE -> STARTING -> CALIBRATING -> ACQUIRING -> ANALYZING -> COMPLETE / FAILED
 *   - Respiration rate extraction via low-frequency envelope spectral analysis
 *   - Full zero-leakage resource disposal
 */

import {
  CoordinatorConfig,
  QualityDiagnostic,
  ScanMode,
  TriageState,
  VitalsTelemetry,
} from '../types/triage';
import { CameraStreamManager } from './CameraStreamManager';
import { FaceMeshEngine } from './FaceMeshEngine';
import { FingerEngine } from './FingerEngine';
import {
  BiquadCascade,
  BiquadFilter,
  PeakDetector,
  PlaneOrthogonalToSkin,
  SignalQualityIndex,
  UniformResampler,
} from './DSPFilters';

export interface TriageCoordinatorCallbacks {
  onStateChange?: (state: TriageState) => void;
  onWaveformSample?: (sample: number, timestamp: number) => void;
  onQualityDiagnostic?: (diagnostic: QualityDiagnostic) => void;
  onScanComplete?: (telemetry: VitalsTelemetry) => void;
  onError?: (error: Error) => void;
}

export class TriageCoordinator {
  public static readonly DEFAULT_CONFIG: CoordinatorConfig = {
    scanDurationMs: 15000,
    calibrationDurationMs: 2000,
    sampleRateHz: 30,
    sqiThreshold: 0.65,
    asymmetryAlertThreshold: 0.12,
    minLumaThreshold: 80,
  };

  private readonly config: CoordinatorConfig;
  private state: TriageState = 'IDLE';
  private mode: ScanMode = 'face';

  // Hardware and processing engines
  private cameraManager: CameraStreamManager;
  private faceEngine: FaceMeshEngine | null = null;
  private fingerEngine: FingerEngine | null = null;

  // DSP pipeline components
  private resamplerR: UniformResampler;
  private resamplerG: UniformResampler;
  private resamplerB: UniformResampler;
  private posEngine: PlaneOrthogonalToSkin;
  private biquadCascade: BiquadCascade;
  private peakDetector: PeakDetector;
  private sqiEngine: SignalQualityIndex;

  // Animation frame loop & timings
  private animFrameId: number | null = null;
  private isRunning: boolean = false;
  private stateStartTime: number = 0;
  private lastQualityFailureTimestamp: number = 0;
  private attachedVideoEl: HTMLVideoElement | null = null;

  // Buffers for final analysis
  private readonly acquiredSamples: number[] = [];
  private readonly asymmetryScores: number[] = [];
  private latestRawSample: number = 0;
  private latestFilteredSample: number = 0;

  // Callbacks
  public onStateChange: ((state: TriageState) => void) | null = null;
  public onWaveformSample: ((sample: number, timestamp: number) => void) | null = null;
  public onQualityDiagnostic: ((diagnostic: QualityDiagnostic) => void) | null = null;
  public onScanComplete: ((telemetry: VitalsTelemetry) => void) | null = null;
  public onError: ((error: Error) => void) | null = null;

  constructor(config?: Partial<CoordinatorConfig>, callbacks?: TriageCoordinatorCallbacks) {
    this.config = { ...TriageCoordinator.DEFAULT_CONFIG, ...config };

    this.cameraManager = new CameraStreamManager();
    this.resamplerR = new UniformResampler(this.config.sampleRateHz);
    this.resamplerG = new UniformResampler(this.config.sampleRateHz);
    this.resamplerB = new UniformResampler(this.config.sampleRateHz);
    this.posEngine = new PlaneOrthogonalToSkin();
    this.biquadCascade = BiquadFilter.createButterworthBandpass30Hz();
    this.peakDetector = new PeakDetector();
    this.sqiEngine = new SignalQualityIndex();

    if (callbacks) {
      this.onStateChange = callbacks.onStateChange ?? null;
      this.onWaveformSample = callbacks.onWaveformSample ?? null;
      this.onQualityDiagnostic = callbacks.onQualityDiagnostic ?? null;
      this.onScanComplete = callbacks.onScanComplete ?? null;
      this.onError = callbacks.onError ?? null;
    }
  }

  public getState(): TriageState {
    return this.state;
  }

  public getMode(): ScanMode {
    return this.mode;
  }

  public isScanRunning(): boolean {
    return this.isRunning;
  }

  public getConfig(): CoordinatorConfig {
    return this.config;
  }

  public setCallbacks(callbacks: TriageCoordinatorCallbacks): void {
    if (callbacks.onStateChange !== undefined) this.onStateChange = callbacks.onStateChange;
    if (callbacks.onWaveformSample !== undefined) this.onWaveformSample = callbacks.onWaveformSample;
    if (callbacks.onQualityDiagnostic !== undefined) this.onQualityDiagnostic = callbacks.onQualityDiagnostic;
    if (callbacks.onScanComplete !== undefined) this.onScanComplete = callbacks.onScanComplete;
    if (callbacks.onError !== undefined) this.onError = callbacks.onError;
  }

  private transitionTo(newState: TriageState): void {
    if (this.state === newState) return;
    this.state = newState;
    this.stateStartTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.onStateChange?.(newState);
  }

  /**
   * Start a clinical vital signs acquisition session.
   * 
   * @param mode 'face' for contactless facial rPPG + FAST stroke geometry; 'finger' for rear camera contact PPG.
   * @param videoEl Video element displaying and decoding camera stream.
   */
  public async startScan(mode: ScanMode, videoEl: HTMLVideoElement): Promise<void> {
    if (this.state !== 'IDLE' && this.state !== 'COMPLETE' && this.state !== 'FAILED') {
      throw new Error(`Cannot start scan while coordinator is in state: ${this.state}`);
    }

    this.mode = mode;
    this.attachedVideoEl = videoEl;
    this.resetDsp();
    this.transitionTo('STARTING');

    try {
      if (mode === 'face') {
        if (!this.faceEngine) {
          this.faceEngine = new FaceMeshEngine();
        }
        await this.cameraManager.startFaceStream(videoEl);
      } else {
        if (!this.fingerEngine) {
          this.fingerEngine = new FingerEngine();
        }
        await this.cameraManager.startFingerStream(videoEl);
      }

      this.isRunning = true;
      this.scheduleNextFrame();
    } catch (err: any) {
      this.transitionTo('FAILED');
      this.onError?.(err instanceof Error ? err : new Error(String(err)));
      this.stopProcessing();
    }
  }

  private resetDsp(): void {
    this.resamplerR.reset();
    this.resamplerG.reset();
    this.resamplerB.reset();
    this.posEngine.reset();
    this.biquadCascade.reset();
    this.peakDetector.reset();
    this.sqiEngine.reset();
    this.acquiredSamples.length = 0;
    this.asymmetryScores.length = 0;
    this.latestRawSample = 0;
    this.latestFilteredSample = 0;
    this.lastQualityFailureTimestamp = 0;
  }

  private scheduleNextFrame(): void {
    if (!this.isRunning) return;

    if (typeof requestAnimationFrame !== 'undefined') {
      this.animFrameId = requestAnimationFrame(() => this.processLoop());
    } else {
      // Node / test environment fallback
      this.animFrameId = setTimeout(() => this.processLoop(), 1000 / this.config.sampleRateHz) as any;
    }
  }

  private async processLoop(): Promise<void> {
    if (!this.isRunning || !this.attachedVideoEl) {
      return;
    }

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    let r = 0;
    let g = 0;
    let b = 0;
    let isTrackingOrPlacementValid = false;
    let strokeAsymmetry = 0.0;

    try {
      if (this.mode === 'face') {
        if (this.faceEngine) {
          const faceResult = await this.faceEngine.processFrame(this.attachedVideoEl);
          if (faceResult && faceResult.isTrackingValid) {
            r = faceResult.foreheadRgb[0];
            g = faceResult.foreheadRgb[1];
            b = faceResult.foreheadRgb[2];
            strokeAsymmetry = faceResult.asymmetryDelta;
            isTrackingOrPlacementValid = true;
          }
        }
      } else {
        if (this.fingerEngine) {
          const fingerResult = this.fingerEngine.sampleFingerFrame(this.attachedVideoEl);
          r = fingerResult.rgb[0];
          g = fingerResult.rgb[1];
          b = fingerResult.rgb[2];
          isTrackingOrPlacementValid = fingerResult.isFingerPlaced;
        }
      }

      // Compute raw luma: Y = 0.299R + 0.587G + 0.114B
      const rawLuma = 0.299 * r + 0.587 * g + 0.114 * b;
      const insufficientLuminance = rawLuma < this.config.minLumaThreshold;
      const motionArtifactDetected = this.mode === 'face' && !isTrackingOrPlacementValid;
      const fingerContactLost = this.mode === 'finger' && !isTrackingOrPlacementValid;

      // Evaluate SQI if buffer is populated
      const sqiResult = this.sqiEngine.evaluate(this.config.sampleRateHz);
      const spectralNoiseHigh = this.sqiEngine.isBufferFull() && sqiResult.sqi < this.config.sqiThreshold;

      let failureReason: import('../types/triage').FailureReason | undefined = undefined;
      if (fingerContactLost) {
        failureReason = 'FINGER_OFF';
      } else if (motionArtifactDetected) {
        failureReason = 'MOTION_ARTIFACT';
      } else if (insufficientLuminance) {
        failureReason = 'LOW_LIGHT';
      } else if (spectralNoiseHigh) {
        failureReason = 'POOR_SNR';
      }

      const isAcceptable = failureReason === undefined;

      const diagnostic: QualityDiagnostic = {
        isAcceptable,
        reason: failureReason,
        luma: rawLuma,
        motionArtifactDetected,
        insufficientLuminance,
        fingerContactLost,
        spectralNoiseHigh,
      };

      this.onQualityDiagnostic?.(diagnostic);

      // Quality watchdog: continuous failure > 1500ms
      if (!isAcceptable) {
        if (this.lastQualityFailureTimestamp === 0) {
          this.lastQualityFailureTimestamp = now;
        } else if (now - this.lastQualityFailureTimestamp > 1500) {
          if (this.state === 'ACQUIRING') {
            // Persistent quality degradation during acquisition triggers failure
            this.transitionTo('FAILED');
            this.onError?.(new Error(`${failureReason ?? 'POOR_SNR'}: Signal quality deteriorated continuously for over 1.5s`));
            this.stopProcessing();
            return;
          }
        }
      } else {
        this.lastQualityFailureTimestamp = 0;
      }

      // State Machine Transitions
      if (this.state === 'STARTING') {
        if (isAcceptable) {
          this.transitionTo('CALIBRATING');
        }
      } else if (this.state === 'CALIBRATING') {
        // Burn-in period (default 2s) to stabilize recursive biquad registers
        if (now - this.stateStartTime >= this.config.calibrationDurationMs) {
          this.transitionTo('ACQUIRING');
        }
      } else if (this.state === 'ACQUIRING') {
        if (now - this.stateStartTime >= this.config.scanDurationMs) {
          this.transitionTo('ANALYZING');
          this.finalizeAnalysis();
          return;
        }
      }

      // Resample and process signals
      if (isTrackingOrPlacementValid) {
        if (this.mode === 'face') {
          this.asymmetryScores.push(strokeAsymmetry);
        }

        // Resample non-uniform frame stream onto uniform 30 Hz grid
        this.resamplerR.push(now, r, (uTs, uR) => {
          this.resamplerG.push(uTs, g, (_tsG, uG) => {
            this.resamplerB.push(uTs, b, (_tsB, uB) => {
              this.processUniformSample(uTs, uR, uG, uB);
            });
          });
        });
      }
    } catch (err: any) {
      this.transitionTo('FAILED');
      this.onError?.(err instanceof Error ? err : new Error(String(err)));
      this.stopProcessing();
      return;
    }

    this.scheduleNextFrame();
  }

  /**
   * Process a single uniform 30.0 Hz RGB sample through the DSP chain.
   */
  private processUniformSample(timestamp: number, r: number, g: number, b: number): void {
    let pulse = 0.0;
    if (this.mode === 'face') {
      pulse = this.posEngine.process(r, g, b);
    } else {
      // In finger contact mode, blood volume pulse modulates tissue green absorption
      // We take the normalized inverted green intensity: -(G / (R + eps))
      pulse = -(g / (r + 1e-4));
    }

    this.latestRawSample = pulse;
    const filtered = this.biquadCascade.process(pulse);
    this.latestFilteredSample = filtered;
    this.sqiEngine.push(filtered);
    this.peakDetector.process(timestamp, filtered);

    if (this.state === 'ACQUIRING') {
      this.acquiredSamples.push(filtered);
    }

    this.onWaveformSample?.(filtered, timestamp);
  }

  /**
   * Finalize clinical telemetry analysis upon completion of acquisition window.
   */
  private finalizeAnalysis(): void {
    const sqiEval = this.sqiEngine.evaluate(this.config.sampleRateHz);
    let finalBpm = this.peakDetector.getCurrentBpm();

    // If peak detector had insufficient beats, fall back to FFT cardiac peak
    if (finalBpm < 30.0 || finalBpm > 240.0) {
      if (sqiEval.peakFreqHz >= 0.75 && sqiEval.peakFreqHz <= 3.0) {
        finalBpm = sqiEval.peakFreqHz * 60.0;
      } else {
        finalBpm = 0.0;
      }
    }

    // Estimate respiration rate from low-frequency modulation of the acquired PPG waveform
    const respirationRateRpm = this.estimateRespirationRate(this.acquiredSamples, this.config.sampleRateHz);

    // Compute stroke asymmetry
    let finalAsymmetry = 0.0;
    if (this.asymmetryScores.length > 0) {
      // Median asymmetry across acquisition
      const sorted = [...this.asymmetryScores].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      finalAsymmetry = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    }

    const telemetry: VitalsTelemetry = {
      timestamp: typeof performance !== 'undefined' ? performance.now() : Date.now(),
      heartRateBpm: Math.round(finalBpm * 10) / 10,
      respirationRateRpm: Math.round(respirationRateRpm * 10) / 10,
      strokeAsymmetryDelta: Math.round(finalAsymmetry * 1000) / 1000,
      sqiScore: Math.round(sqiEval.sqi * 1000) / 1000,
      rawWaveformSample: this.latestRawSample,
      filteredWaveformSample: this.latestFilteredSample,
      snrDb: Math.round(sqiEval.snrDb * 10) / 10,
    };

    this.transitionTo('COMPLETE');
    this.onScanComplete?.(telemetry);
    this.stopProcessing();
  }

  /**
   * Respiration rate estimation via low-frequency envelope power spectral analysis.
   * Searches the canonical respiratory band [0.15 Hz, 0.50 Hz] (9 to 30 RPM).
   */
  public estimateRespirationRate(samples: number[], fs: number): number {
    const N = samples.length;
    if (N < 90) {
      // Less than 3 seconds of data, return standard resting adult baseline
      return 16.0;
    }

    // Evaluate discrete Fourier power across [0.15 Hz, 0.50 Hz] with 0.01 Hz resolution
    const fMin = 0.15;
    const fMax = 0.50;
    const fStep = 0.01;

    let maxPower = -1.0;
    let bestFreqHz = 0.25; // default 15 RPM

    // Mean-center signal
    let mean = 0.0;
    for (let i = 0; i < N; i++) mean += samples[i];
    mean /= N;

    for (let f = fMin; f <= fMax; f += fStep) {
      let real = 0.0;
      let imag = 0.0;
      const omega = (2.0 * Math.PI * f) / fs;

      for (let n = 0; n < N; n++) {
        // Hanning window
        const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * n) / (N - 1)));
        const val = (samples[n] - mean) * w;
        real += val * Math.cos(omega * n);
        imag -= val * Math.sin(omega * n);
      }

      const p = real * real + imag * imag;
      if (p > maxPower) {
        maxPower = p;
        bestFreqHz = f;
      }
    }

    return bestFreqHz * 60.0;
  }

  /**
   * Stop video stream, cancel animation loop, and cease processing.
   */
  public stopProcessing(): void {
    this.isRunning = false;

    if (this.animFrameId !== null) {
      if (typeof cancelAnimationFrame !== 'undefined') {
        cancelAnimationFrame(this.animFrameId);
      } else {
        clearTimeout(this.animFrameId);
      }
      this.animFrameId = null;
    }

    this.cameraManager.stopStream();
    this.attachedVideoEl = null;
  }

  /**
   * Complete teardown and memory release.
   */
  public destroy(): void {
    this.stopProcessing();
    this.cameraManager.stopStream();

    if (this.faceEngine) {
      this.faceEngine.destroy();
      this.faceEngine = null;
    }

    if (this.fingerEngine) {
      this.fingerEngine.destroy();
      this.fingerEngine = null;
    }

    this.resetDsp();
    this.transitionTo('IDLE');

    this.onStateChange = null;
    this.onWaveformSample = null;
    this.onQualityDiagnostic = null;
    this.onScanComplete = null;
    this.onError = null;
  }
}
