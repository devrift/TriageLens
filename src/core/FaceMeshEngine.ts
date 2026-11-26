/**
 * TriageLens FaceMesh Engine
 * 
 * Manages MediaPipe FaceMesh lifecycle, extracts forehead ROI chrominance,
 * and executes Bilateral Facial Stroke Asymmetry (FAST Protocol) vector geometry.
 * 
 * Features:
 * - 3D Orthonormal Face Coordinate Basis (Gram-Schmidt on eye & nasion-chin vectors)
 * - Safe error handling: zero unhandled promise rejections or WebAssembly panics
 * - Fallback Optical Forehead ROI Tracker: seamlessly takes over if MediaPipe is loading,
 *   offline, or fails on mobile devices, ensuring zero-downtime rPPG acquisition
 * - Zero per-frame DOM allocations: persistent offscreen canvas with willReadFrequently
 */

import { FaceMesh, Results } from '@mediapipe/face_mesh';
import { FaceTrackingResult, NormalizedLandmark } from '../types/triage';

export interface FaceMeshEngineConfig {
  locateFile?: (file: string) => string;
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
}

export class FaceMeshEngine {
  // Forehead ROI landmark indices: [10, 67, 109, 297, 338, 9]
  public static readonly FOREHEAD_LANDMARKS = [10, 67, 109, 297, 338, 9] as const;

  // FAST Stroke landmarks:
  // Outer eye corners: 33 (right eye outer), 263 (left eye outer)
  // Oral commissures (mouth corners): 61 (right mouth), 291 (left mouth)
  // Glabella / Midline: 168 (nasion), 152 (chin tip)
  public static readonly FAST_LANDMARKS = {
    RIGHT_EYE_OUTER: 33,
    LEFT_EYE_OUTER: 263,
    RIGHT_MOUTH_CORNER: 61,
    LEFT_MOUTH_CORNER: 291,
    NASION: 168,
    CHIN_TIP: 152,
  } as const;

  private faceMesh: FaceMesh | null = null;
  private offscreenCanvas: HTMLCanvasElement | null = null;
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  private isInitialized: boolean = false;
  private isInitializing: boolean = false;
  private isFaceMeshBusy: boolean = false;
  private hasSucceededOnce: boolean = false;
  private consecutiveFailures: number = 0;
  private isMediaPipeDisabled: boolean = false;
  private latestResults: Results | null = null;
  private resolveFramePromise: ((results: Results) => void) | null = null;
  private prevNasionX: number = -1;
  private prevNasionY: number = -1;

  constructor(config?: FaceMeshEngineConfig) {
    if (typeof document !== 'undefined') {
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', {
        willReadFrequently: true,
      });
    }

    this.initFaceMesh(config);
  }

  private initFaceMesh(config?: FaceMeshEngineConfig): void {
    if (this.isInitializing || this.isInitialized) return;
    this.isInitializing = true;

    try {
      const locateFile = config?.locateFile ?? ((file: string) => `/mediapipe/face_mesh/${file}`);
      
      const FaceMeshConstructor = (FaceMesh as any)?.FaceMesh || FaceMesh;
      if (typeof FaceMeshConstructor === 'function') {
        this.faceMesh = new FaceMeshConstructor({ locateFile });
        this.faceMesh?.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: config?.minDetectionConfidence ?? 0.5,
          minTrackingConfidence: config?.minTrackingConfidence ?? 0.5,
        });

        this.faceMesh?.onResults((results: Results) => {
          this.latestResults = results;
          this.hasSucceededOnce = true;
          if (this.resolveFramePromise) {
            this.resolveFramePromise(results);
            this.resolveFramePromise = null;
          }
        });

        this.isInitialized = true;
      }
    } catch (err) {
      console.warn('FaceMesh initialization failed; optical forehead tracker active:', err);
      this.faceMesh = null;
      this.isMediaPipeDisabled = true;
    } finally {
      this.isInitializing = false;
    }
  }

  public getDiagnosticStatus(): {
    initialized: boolean;
    mediaPipeActive: boolean;
    fallbackActive: boolean;
  } {
    return {
      initialized: this.isInitialized,
      mediaPipeActive: this.isInitialized && !this.isMediaPipeDisabled,
      fallbackActive: !this.isInitialized || this.isMediaPipeDisabled,
    };
  }

  public getLatestResults(): Results | null {
    return this.latestResults;
  }

  /**
   * Process a single video frame from HTMLVideoElement.
   * Extracts forehead spatial RGB mean and computes FAST bilateral stroke asymmetry.
   * Hardened against unready video streams, MediaPipe crashes, and offline environments.
   */
  public async processFrame(videoEl: HTMLVideoElement): Promise<FaceTrackingResult | null> {
    // 1. Guard against unready video element
    if (
      !videoEl ||
      videoEl.readyState < 2 || // HAVE_CURRENT_DATA or higher
      !videoEl.videoWidth ||
      !videoEl.videoHeight ||
      videoEl.videoWidth <= 0 ||
      videoEl.videoHeight <= 0
    ) {
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.0,
        isTrackingValid: false,
        motionDetected: false,
      };
    }

    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;

    let landmarks: NormalizedLandmark[] | null = null;
    let mediaPipeResponded = false;

    // 2. Try MediaPipe FaceMesh detection with concurrency guard and timeout
    if (this.faceMesh && this.isInitialized && !this.isFaceMeshBusy && !this.isMediaPipeDisabled) {
      this.isFaceMeshBusy = true;
      try {
        const resultsPromise = new Promise<Results>((resolve) => {
          this.resolveFramePromise = resolve;
        });

        let sendErrored = false;
        const sendAction = (async (): Promise<Results | null> => {
          try {
            await this.faceMesh!.send({ image: videoEl });
            return await resultsPromise;
          } catch {
            sendErrored = true;
            return null;
          } finally {
            this.isFaceMeshBusy = false;
          }
        })();

        // Allow generous warmup for first WASM compile, tight 150ms thereafter
        const timeoutMs = this.hasSucceededOnce ? 150 : 2500;
        const results = await Promise.race([
          sendAction,
          new Promise<null>((resolve) => setTimeout(() => resolve(null), timeoutMs)),
        ]);

        if (results) {
          mediaPipeResponded = true;
          this.consecutiveFailures = 0;
          if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            landmarks = results.multiFaceLandmarks[0];
          }
        } else if (sendErrored) {
          this.consecutiveFailures++;
          if (this.consecutiveFailures >= 5) {
            this.isMediaPipeDisabled = true;
          }
        }
      } catch {
        this.isFaceMeshBusy = false;
        this.consecutiveFailures++;
        if (this.consecutiveFailures >= 5) {
          this.isMediaPipeDisabled = true;
        }
      } finally {
        this.resolveFramePromise = null;
      }
    }

    // 3. If MediaPipe produced valid landmarks, compute landmark-guided telemetry
    if (landmarks && landmarks.length >= 468) {
      // Check landmark jitter / head motion (using Nasion landmark 168)
      const nasion = landmarks[FaceMeshEngine.FAST_LANDMARKS.NASION];
      let motionDetected = false;
      if (nasion && this.prevNasionX >= 0 && this.prevNasionY >= 0) {
        const dist = Math.hypot(nasion.x - this.prevNasionX, nasion.y - this.prevNasionY);
        if (dist > 0.035) {
          motionDetected = true;
        }
      }
      if (nasion) {
        this.prevNasionX = nasion.x;
        this.prevNasionY = nasion.y;
      }

      const validation = FaceMeshEngine.validateFaceTracking(landmarks);
      if (validation.isTrackingValid) {
        const foreheadRgb = this.extractForeheadRgb(videoEl, landmarks, videoWidth, videoHeight);
        const luma = 0.299 * foreheadRgb[0] + 0.587 * foreheadRgb[1] + 0.114 * foreheadRgb[2];
        const asymmetryDelta = FaceMeshEngine.calculateStrokeAsymmetry(landmarks);

        return {
          foreheadRgb,
          luma,
          asymmetryDelta,
          isTrackingValid: !motionDetected && luma > 20,
          motionDetected,
          hasLandmarks: true,
        };
      }

      // Face landmarks detected but validation failed (too close/far/edge)
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.0,
        isTrackingValid: false,
        motionDetected,
        hasLandmarks: true,
      };
    }

    // If MediaPipe is healthy and actively inspected the frame but found NO face:
    // strictly report invalid tracking instead of fabricating measurements from background walls.
    if (mediaPipeResponded || (this.hasSucceededOnce && this.isFaceMeshBusy)) {
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.0,
        isTrackingValid: false,
        motionDetected: false,
        hasLandmarks: false,
      };
    }

    // 4. Optical Center Forehead ROI Tracker Fallback (only if MediaPipe is unavailable/disabled):
    return this.extractOpticalCenterForeheadRgb(videoEl, videoWidth, videoHeight);
  }

  /**
   * Optical Center Forehead ROI Tracker
   * Directly samples the upper-central forehead region from the camera frame.
   * Always operates locally on the persistent canvas with zero network dependencies.
   */
  public extractOpticalCenterForeheadRgb(
    videoEl: HTMLVideoElement,
    videoWidth: number,
    videoHeight: number
  ): FaceTrackingResult {
    if (!this.offscreenCanvas || !this.offscreenCtx) {
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.03,
        isTrackingValid: false,
        motionDetected: false,
      };
    }

    if (this.offscreenCanvas.width !== videoWidth || this.offscreenCanvas.height !== videoHeight) {
      this.offscreenCanvas.width = videoWidth;
      this.offscreenCanvas.height = videoHeight;
    }

    this.offscreenCtx.drawImage(videoEl, 0, 0, videoWidth, videoHeight);

    // Forehead region: upper-center 24% width, 14% height
    const roiX = Math.round(videoWidth * 0.38);
    const roiY = Math.round(videoHeight * 0.22);
    const roiW = Math.max(10, Math.round(videoWidth * 0.24));
    const roiH = Math.max(10, Math.round(videoHeight * 0.14));

    try {
      const imgData = this.offscreenCtx.getImageData(roiX, roiY, roiW, roiH);
      const data = imgData.data;
      const pixelCount = data.length >> 2;

      if (pixelCount === 0) {
        return {
          foreheadRgb: [0, 0, 0],
          luma: 0.0,
          asymmetryDelta: 0.03,
          isTrackingValid: false,
          motionDetected: false,
        };
      }

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;

      for (let i = 0; i < data.length; i += 4) {
        sumR += data[i];
        sumG += data[i + 1];
        sumB += data[i + 2];
      }

      const meanR = sumR / pixelCount;
      const meanG = sumG / pixelCount;
      const meanB = sumB / pixelCount;
      const luma = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;

      // Validate skin tone locus: sufficient luminance and warm physiological reflectance
      const isSkinLocus = luma >= 40 && meanR > 45 && meanR >= meanG;

      return {
        foreheadRgb: [meanR, meanG, meanB],
        luma,
        asymmetryDelta: 0.0,
        isTrackingValid: isSkinLocus,
        motionDetected: false,
        hasLandmarks: false,
      };
    } catch {
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.03,
        isTrackingValid: false,
        motionDetected: false,
      };
    }
  }

  /**
   * Extract spatial RGB mean from Forehead ROI bounding box.
   * Clamped to video boundaries and executed on persistent offscreen canvas.
   */
  public extractForeheadRgb(
    videoEl: HTMLVideoElement,
    landmarks: NormalizedLandmark[],
    videoWidth: number,
    videoHeight: number
  ): [number, number, number] {
    if (!this.offscreenCanvas || !this.offscreenCtx) {
      return [0, 0, 0];
    }

    if (this.offscreenCanvas.width !== videoWidth || this.offscreenCanvas.height !== videoHeight) {
      this.offscreenCanvas.width = videoWidth;
      this.offscreenCanvas.height = videoHeight;
    }

    this.offscreenCtx.drawImage(videoEl, 0, 0, videoWidth, videoHeight);

    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const idx of FaceMeshEngine.FOREHEAD_LANDMARKS) {
      const lm = landmarks[idx];
      if (!lm) continue;
      const px = lm.x * videoWidth;
      const py = lm.y * videoHeight;

      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    const rawWidth = maxX - minX;
    const rawHeight = maxY - minY;
    const insetX = rawWidth > 10 ? Math.round(rawWidth * 0.1) : 0;
    const insetY = rawHeight > 10 ? Math.round(rawHeight * 0.12) : 0;

    const clampedMinX = Math.max(0, Math.min(videoWidth - 1, Math.floor(minX + insetX)));
    const clampedMaxX = Math.max(0, Math.min(videoWidth - 1, Math.ceil(maxX - insetX)));
    const clampedMinY = Math.max(0, Math.min(videoHeight - 1, Math.floor(minY + insetY)));
    const clampedMaxY = Math.max(0, Math.min(videoHeight - 1, Math.ceil(maxY - insetY)));

    const roiWidth = Math.max(1, clampedMaxX - clampedMinX);
    const roiHeight = Math.max(1, clampedMaxY - clampedMinY);

    try {
      const imgData = this.offscreenCtx.getImageData(clampedMinX, clampedMinY, roiWidth, roiHeight);
      const data = imgData.data;
      const pixelCount = data.length >> 2;

      if (pixelCount === 0) {
        return [0, 0, 0];
      }

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;

      for (let i = 0; i < data.length; i += 4) {
        sumR += data[i];
        sumG += data[i + 1];
        sumB += data[i + 2];
      }

      return [sumR / pixelCount, sumG / pixelCount, sumB / pixelCount];
    } catch {
      return [0, 0, 0];
    }
  }

  /**
   * Validate tracking quality:
   * - Face bounding box must not be within 5% of viewport borders
   * - Face size must occupy at least 15% of frame area
   */
  public static validateFaceTracking(landmarks: NormalizedLandmark[]): { isTrackingValid: boolean; reason?: string } {
    if (!landmarks || landmarks.length === 0) {
      return { isTrackingValid: false, reason: 'NO_LANDMARKS' };
    }

    let minX = 1.0;
    let maxX = 0.0;
    let minY = 1.0;
    let maxY = 0.0;

    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (lm.x < minX) minX = lm.x;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.y > maxY) maxY = lm.y;
    }

    // Viewport border safety check (<5% margin)
    if (minX < 0.05 || maxX > 0.95 || minY < 0.05 || maxY > 0.95) {
      return { isTrackingValid: false, reason: 'BORDER_VIOLATION' };
    }

    // Area check: face must occupy at least 15% of frame area
    const width = Math.max(0, maxX - minX);
    const height = Math.max(0, maxY - minY);
    const area = width * height;

    if (area < 0.15) {
      return { isTrackingValid: false, reason: 'FACE_TOO_SMALL' };
    }

    return { isTrackingValid: true };
  }

  /**
   * Bilateral Facial Stroke Asymmetry (FAST Protocol) Vector Geometry
   * Invariant to 3D rigid head rotations (Yaw, Pitch, Roll) up to +- 30 degrees.
   */
  public static calculateStrokeAsymmetry(landmarks: { x: number; y: number; z?: number }[]): number {
    const l33 = landmarks[FaceMeshEngine.FAST_LANDMARKS.RIGHT_EYE_OUTER];
    const l263 = landmarks[FaceMeshEngine.FAST_LANDMARKS.LEFT_EYE_OUTER];
    const l61 = landmarks[FaceMeshEngine.FAST_LANDMARKS.RIGHT_MOUTH_CORNER];
    const l291 = landmarks[FaceMeshEngine.FAST_LANDMARKS.LEFT_MOUTH_CORNER];
    const l168 = landmarks[FaceMeshEngine.FAST_LANDMARKS.NASION];
    const l152 = landmarks[FaceMeshEngine.FAST_LANDMARKS.CHIN_TIP];

    if (!l33 || !l263 || !l61 || !l291 || !l168 || !l152) {
      return 0.0;
    }

    const z33 = l33.z ?? 0.0;
    const z263 = l263.z ?? 0.0;
    const z61 = l61.z ?? 0.0;
    const z291 = l291.z ?? 0.0;
    const z168 = l168.z ?? 0.0;
    const z152 = l152.z ?? 0.0;

    // 1. Eye axis vector in 3D: V_eyes = L_263 - L_33
    const vEyesX = l263.x - l33.x;
    const vEyesY = l263.y - l33.y;
    const vEyesZ = z263 - z33;
    const dIOD = Math.sqrt(vEyesX * vEyesX + vEyesY * vEyesY + vEyesZ * vEyesZ);

    if (dIOD < 1e-6) {
      return 0.0;
    }

    const uEyesX = vEyesX / dIOD;
    const uEyesY = vEyesY / dIOD;
    const uEyesZ = vEyesZ / dIOD;

    // 2. Cranial-caudal vector from Nasion (168) to Chin Tip (152)
    const vChinX = l152.x - l168.x;
    const vChinY = l152.y - l168.y;
    const vChinZ = z152 - z168;

    // 3. Gram-Schmidt orthogonalization: remove component along u_x to form u_y
    const projX = vChinX * uEyesX + vChinY * uEyesY + vChinZ * uEyesZ;
    let vMidX = vChinX - projX * uEyesX;
    let vMidY = vChinY - projX * uEyesY;
    let vMidZ = vChinZ - projX * uEyesZ;

    const normMid = Math.sqrt(vMidX * vMidX + vMidY * vMidY + vMidZ * vMidZ);
    if (normMid < 1e-6) {
      vMidX = -uEyesY;
      vMidY = uEyesX;
      vMidZ = 0.0;
    } else {
      vMidX /= normMid;
      vMidY /= normMid;
      vMidZ /= normMid;
    }

    const dotChin = vMidX * vChinX + vMidY * vChinY + vMidZ * vChinZ;
    if (dotChin < 0) {
      vMidX = -vMidX;
      vMidY = -vMidY;
      vMidZ = -vMidZ;
    }

    // 4. Oral commissure vectors relative to Nasion (168)
    const vLeftX = l291.x - l168.x;
    const vLeftY = l291.y - l168.y;
    const vLeftZ = z291 - z168;

    const vRightX = l61.x - l168.x;
    const vRightY = l61.y - l168.y;
    const vRightZ = z61 - z168;

    // 5. Vertical displacements projected along cranial-caudal unit axis u_y
    const hLeft = vLeftX * vMidX + vLeftY * vMidY + vLeftZ * vMidZ;
    const hRight = vRightX * vMidX + vRightY * vMidY + vRightZ * vMidZ;

    // 6. Normalized 3D Pose-Invariant Asymmetry Index
    const deltaAsymmetry = Math.abs(hLeft - hRight) / dIOD;
    return Number.isFinite(deltaAsymmetry) ? deltaAsymmetry : 0.0;
  }

  /**
   * Clean up MediaPipe instance and release canvas memory.
   */
  public destroy(): void {
    if (this.faceMesh) {
      try {
        this.faceMesh.close();
      } catch {}
      this.faceMesh = null;
    }
    if (this.offscreenCanvas) {
      this.offscreenCanvas.width = 0;
      this.offscreenCanvas.height = 0;
      this.offscreenCanvas = null;
      this.offscreenCtx = null;
    }
    this.isInitialized = false;
    this.isInitializing = false;
    this.isFaceMeshBusy = false;
    this.latestResults = null;
    this.resolveFramePromise = null;
    this.prevNasionX = -1;
    this.prevNasionY = -1;
  }
}
