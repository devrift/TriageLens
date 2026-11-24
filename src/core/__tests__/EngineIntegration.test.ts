import { describe, it, expect, vi } from 'vitest';
import { FingerEngine } from '../FingerEngine';
import { FaceMeshEngine } from '../FaceMeshEngine';
import { TriageCoordinator } from '../TriageCoordinator';
import { CameraStreamManager } from '../CameraStreamManager';
import { CameraStreamError } from '../../types/triage';

describe('Engine Integration & Subsystem Tests', () => {
  describe('FingerEngine Contact PPG Gate', () => {
    it('approves capillary transillumination when tissue is saturated in red and absorbing green', () => {
      // True capillary transillumination: R=190, G=50, B=40
      const isValid = FingerEngine.validateContact(190, 50, 40);
      expect(isValid).toBe(true);
    });

    it('rejects low luminance / ambient skin exposure', () => {
      // Ambient skin under normal light: R=110, G=80, B=60 (R <= 130)
      const isValid = FingerEngine.validateContact(110, 80, 60);
      expect(isValid).toBe(false);
    });

    it('rejects white paper or unabsorbed white flash', () => {
      // Overexposed white light: R=245, G=240, B=235 (G >= 100, Ratio ~ 1.02)
      const isValid = FingerEngine.validateContact(245, 240, 235);
      expect(isValid).toBe(false);
    });

    it('rejects green absorption failure where R/G ratio is below 1.8', () => {
      const isValid = FingerEngine.validateContact(150, 95, 80); // 150/95 = 1.57 < 1.8
      expect(isValid).toBe(false);
    });
  });

  describe('FaceMeshEngine Tracking Validation', () => {
    function generateSyntheticFace(
      minX: number,
      maxX: number,
      minY: number,
      maxY: number
    ): { x: number; y: number; z: number }[] {
      const landmarks: { x: number; y: number; z: number }[] = [];
      for (let i = 0; i < 468; i++) {
        // Distribute within specified box
        const frac = i / 468;
        landmarks.push({
          x: minX + frac * (maxX - minX),
          y: minY + frac * (maxY - minY),
          z: 0.0,
        });
      }
      return landmarks;
    }

    it('rejects face too close to viewport border (<5% border margin)', () => {
      // Face touches left border at x=0.02
      const landmarks = generateSyntheticFace(0.02, 0.60, 0.20, 0.70);
      const val = FaceMeshEngine.validateFaceTracking(landmarks);
      expect(val.isTrackingValid).toBe(false);
      expect(val.reason).toBe('BORDER_VIOLATION');
    });

    it('rejects face occupying < 15% of frame area', () => {
      // Box 0.3 x 0.3 = 0.09 area (< 0.15)
      const landmarks = generateSyntheticFace(0.35, 0.65, 0.35, 0.65);
      const val = FaceMeshEngine.validateFaceTracking(landmarks);
      expect(val.isTrackingValid).toBe(false);
      expect(val.reason).toBe('FACE_TOO_SMALL');
    });

    it('accepts centered face with sufficient frame area (>15% area and >5% margin)', () => {
      // Box 0.5 x 0.5 = 0.25 area (>= 0.15), margins 0.25 from edges
      const landmarks = generateSyntheticFace(0.25, 0.75, 0.25, 0.75);
      const val = FaceMeshEngine.validateFaceTracking(landmarks);
      expect(val.isTrackingValid).toBe(true);
    });

    it('safely handles unready video elements with zero dimensions without throwing', async () => {
      const engine = new FaceMeshEngine();
      const mockVideo = {
        readyState: 0,
        videoWidth: 0,
        videoHeight: 0,
      } as any;

      const res = await engine.processFrame(mockVideo);
      expect(res).not.toBeNull();
      expect(res?.isTrackingValid).toBe(false);
      expect(res?.foreheadRgb).toEqual([0, 0, 0]);
    });

    it('optical center forehead extractor provides valid fallback without crashing', () => {
      const engine = new FaceMeshEngine();
      // Test direct optical method execution
      const mockVideo = {
        readyState: 4,
        videoWidth: 640,
        videoHeight: 480,
      } as any;

      const res = engine.extractOpticalCenterForeheadRgb(mockVideo, 640, 480);
      expect(res).not.toBeNull();
      expect(typeof res.asymmetryDelta).toBe('number');
      expect(res.asymmetryDelta).toBe(0.03);
    });
  });

  describe('TriageCoordinator Respiration Rate Estimator', () => {
    it('accurately identifies 15 RPM (0.25 Hz) breathing rate from low-frequency envelope', () => {
      const coordinator = new TriageCoordinator();
      const fs = 30.0;
      const durationSec = 15.0; // 15 seconds
      const totalSamples = Math.round(durationSec * fs);

      const targetRpm = 15.0;
      const targetFreqHz = targetRpm / 60.0; // 0.25 Hz

      const samples: number[] = [];
      for (let i = 0; i < totalSamples; i++) {
        const t = i / fs;
        // Respiratory-induced baseline modulation at 0.25 Hz + small cardiac pulse at 1.2 Hz
        const respiratoryModulation = 1.5 * Math.sin(2.0 * Math.PI * targetFreqHz * t);
        const cardiacPulse = 0.5 * Math.sin(2.0 * Math.PI * 1.2 * t);
        samples.push(respiratoryModulation + cardiacPulse);
      }

      const estimatedRpm = coordinator.estimateRespirationRate(samples, fs);
      expect(Math.abs(estimatedRpm - targetRpm)).toBeLessThanOrEqual(1.0);
    });
  });

  describe('CameraStreamManager Typed Errors', () => {
    it('instantiates CameraStreamError with correct code and name', () => {
      const err = new CameraStreamError('CAMERA_PERMISSION_DENIED', 'Permission denied');
      expect(err.code).toBe('CAMERA_PERMISSION_DENIED');
      expect(err.name).toBe('CameraStreamError');
      expect(err instanceof Error).toBe(true);
    });

    it('rejects with CAMERA_NOT_FOUND when mediaDevices is unavailable', async () => {
      const manager = new CameraStreamManager();
      const mockVideoEl = {} as HTMLVideoElement;

      await expect(manager.startFaceStream(mockVideoEl)).rejects.toThrow();
    });
  });

  describe('TriageCoordinator State Machine Lifecycle', () => {
    it('initializes in IDLE state with default configuration', () => {
      const coordinator = new TriageCoordinator();
      expect(coordinator.getState()).toBe('IDLE');
      expect(coordinator.getConfig().scanDurationMs).toBe(15000);
      expect(coordinator.getConfig().calibrationDurationMs).toBe(2000);
      expect(coordinator.getConfig().sampleRateHz).toBe(30);
      expect(coordinator.getConfig().sqiThreshold).toBe(0.65);
      expect(coordinator.getConfig().asymmetryAlertThreshold).toBe(0.12);
      expect(coordinator.getConfig().minLumaThreshold).toBe(80);
    });

    it('cleans up resources on destroy() without errors', () => {
      const coordinator = new TriageCoordinator();
      const onStateChange = vi.fn();
      coordinator.onStateChange = onStateChange;

      coordinator.destroy();
      expect(coordinator.getState()).toBe('IDLE');
    });

    it('enforces 1.5s continuous bad-quality failure watchdog in ACQUIRING state', async () => {
      const coordinator = new TriageCoordinator({
        scanDurationMs: 15000,
        calibrationDurationMs: 100,
        sampleRateHz: 30,
        minLumaThreshold: 80,
      });

      const states: string[] = [];
      coordinator.onStateChange = (st) => states.push(st);

      let capturedError: Error | null = null;
      coordinator.onError = (err) => {
        capturedError = err;
      };

      // Manually invoke state transition to ACQUIRING
      (coordinator as any).transitionTo('ACQUIRING');
      expect(coordinator.getState()).toBe('ACQUIRING');

      // Create a mock video element with pitch black pixels (luma = 0 < 80)
      const mockVideo = {
        videoWidth: 320,
        videoHeight: 240,
        srcObject: null,
        playsInline: true,
        muted: true,
        autoplay: true,
      } as unknown as HTMLVideoElement;

      (coordinator as any).attachedVideoEl = mockVideo;
      (coordinator as any).isRunning = true;
      (coordinator as any).mode = 'finger';

      // Mock finger engine to report unplaced / low luma
      (coordinator as any).fingerEngine = {
        sampleFingerFrame: () => ({
          rgb: [0, 0, 0],
          luma: 0,
          isFingerPlaced: false,
          isPlacementValid: false,
          signalValue: 0,
          timestamp: 1000,
        }),
        destroy: () => {},
      };

      // First frame at t = 1000: initiates failure timer
      const originalPerf = globalThis.performance;
      let mockTime = 1000;
      (globalThis as any).performance = { now: () => mockTime };

      await (coordinator as any).processLoop();
      expect(coordinator.getState()).toBe('ACQUIRING'); // Not yet failed

      // Second frame at t = 2000 (1000ms bad quality elapsed): still within 1.5s window
      mockTime = 2000;
      await (coordinator as any).processLoop();
      expect(coordinator.getState()).toBe('ACQUIRING');

      // Third frame at t = 2600 (1600ms bad quality elapsed > 1500ms): must fail!
      mockTime = 2600;
      await (coordinator as any).processLoop();

      expect(coordinator.getState()).toBe('FAILED');
      expect(capturedError).not.toBeNull();
      expect((capturedError as any)?.message).toContain('FINGER_OFF');

      coordinator.destroy();
      (globalThis as any).performance = originalPerf;
    });
  });

  describe('FaceMeshEngine Landmark Edge Cases', () => {
    it('returns zero asymmetry and safe defaults for missing or empty landmarks', () => {
      const asymmetryEmpty = FaceMeshEngine.calculateStrokeAsymmetry([]);
      expect(asymmetryEmpty).toBe(0.0);

      const trackingEmpty = FaceMeshEngine.validateFaceTracking([]);
      expect(trackingEmpty.isTrackingValid).toBe(false);
      expect(trackingEmpty.reason).toBe('NO_LANDMARKS');
    });

    it('handles degenerate eye landmarks with zero distance without dividing by zero', () => {
      const landmarks: any[] = [];
      for (let i = 0; i < 468; i++) {
        landmarks.push({ x: 0.5, y: 0.5, z: 0.0 });
      }
      const asymmetry = FaceMeshEngine.calculateStrokeAsymmetry(landmarks);
      expect(asymmetry).toBe(0.0);
      expect(Number.isFinite(asymmetry)).toBe(true);
    });
  });
});
