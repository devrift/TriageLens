/**
 * TriageLens FaceMesh Engine
 * 
 * Manages MediaPipe FaceMesh lifecycle, extracts forehead ROI chrominance,
 * and executes Bilateral Facial Stroke Asymmetry (FAST Protocol) vector geometry.
 * 
 * Zero per-frame DOM allocations: single persistent offscreen canvas with willReadFrequently.
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
    const locateFile = config?.locateFile ?? ((file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`);
    
    // Support both default and named export shapes of @mediapipe/face_mesh
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
        if (this.resolveFramePromise) {
          this.resolveFramePromise(results);
          this.resolveFramePromise = null;
        }
      });

      this.isInitialized = true;
    }
  }

  public getLatestResults(): Results | null {
    return this.latestResults;
  }

  /**
   * Process a single video frame from HTMLVideoElement.
   * Extracts forehead spatial RGB mean and computes FAST bilateral stroke asymmetry.
   */
  public async processFrame(videoEl: HTMLVideoElement): Promise<FaceTrackingResult | null> {
    if (!this.faceMesh || !this.isInitialized) {
      return null;
    }

    const videoWidth = videoEl.videoWidth || 640;
    const videoHeight = videoEl.videoHeight || 480;

    // Send frame to MediaPipe pipeline
    const resultsPromise = new Promise<Results>((resolve) => {
      this.resolveFramePromise = resolve;
    });

    await this.faceMesh.send({ image: videoEl });
    const results = await resultsPromise;

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      this.prevNasionX = -1;
      this.prevNasionY = -1;
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.0,
        isTrackingValid: false,
        motionDetected: true,
      };
    }

    const landmarks = results.multiFaceLandmarks[0];

    // Check landmark jitter / head motion (using Nasion landmark 168)
    const nasion = landmarks[FaceMeshEngine.FAST_LANDMARKS.NASION];
    let motionDetected = false;
    if (nasion && this.prevNasionX >= 0 && this.prevNasionY >= 0) {
      const dist = Math.hypot(nasion.x - this.prevNasionX, nasion.y - this.prevNasionY);
      // More than 3% viewport displacement in a single frame indicates abrupt movement
      if (dist > 0.03) {
        motionDetected = true;
      }
    }
    if (nasion) {
      this.prevNasionX = nasion.x;
      this.prevNasionY = nasion.y;
    }

    // 1. Validate face bounding box and size
    const validation = FaceMeshEngine.validateFaceTracking(landmarks);
    if (!validation.isTrackingValid) {
      return {
        foreheadRgb: [0, 0, 0],
        luma: 0.0,
        asymmetryDelta: 0.0,
        isTrackingValid: false,
        motionDetected,
      };
    }

    // 2. Extract Forehead ROI RGB spatial mean
    const foreheadRgb = this.extractForeheadRgb(videoEl, landmarks, videoWidth, videoHeight);
    const luma = 0.299 * foreheadRgb[0] + 0.587 * foreheadRgb[1] + 0.114 * foreheadRgb[2];

    // 3. Compute FAST bilateral facial stroke asymmetry
    const asymmetryDelta = FaceMeshEngine.calculateStrokeAsymmetry(landmarks);

    return {
      foreheadRgb,
      luma,
      asymmetryDelta,
      isTrackingValid: !motionDetected,
      motionDetected,
    };
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

    // Ensure canvas dimensions match video
    if (this.offscreenCanvas.width !== videoWidth || this.offscreenCanvas.height !== videoHeight) {
      this.offscreenCanvas.width = videoWidth;
      this.offscreenCanvas.height = videoHeight;
    }

    // Draw video frame to persistent canvas
    this.offscreenCtx.drawImage(videoEl, 0, 0, videoWidth, videoHeight);

    // Compute bounding box for forehead landmarks: [10, 67, 109, 297, 338, 9]
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

    // Clamp coordinates to video dimensions with safety margins against hair and eyebrows
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

    // Check viewport borders: must not be within 5% (0.05) of edges
    if (minX < 0.05 || maxX > 0.95 || minY < 0.05 || maxY > 0.95) {
      return { isTrackingValid: false, reason: 'BORDER_VIOLATION' };
    }

    // Check face area: (width * height) must be >= 15% (0.15) of total frame area
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
   * 
   * Mathematical derivation in 3D:
   *   1. Outer eye vector in 3D: V_eyes = L_263 (left eye outer) - L_33 (right eye outer)
   *   2. 3D Inter-Ocular Distance: d_IOD = ||V_eyes||_2
   *   3. Normalized lateral unit vector: u_x = V_eyes / d_IOD
   *   4. Cranial-caudal midline vector from Nasion (168) to Chin Tip (152): V_chin = L_152 - L_168
   *   5. Gram-Schmidt orthogonalization for vertical unit vector:
   *        V_mid = V_chin - (V_chin . u_x) * u_x
   *        u_y = V_mid / ||V_mid||_2
   *   6. Mouth corner vectors relative to Nasion:
   *        V_left  = L_291 - L_168
   *        V_right = L_61  - L_168
   *   7. Vertical cranial-caudal displacements projected along u_y:
   *        h_left  = V_left  . u_y
   *        h_right = V_right . u_y
   *   8. Normalized 3D Pose-Invariant Asymmetry Index:
   *        Delta_asymmetry = |h_left - h_right| / d_IOD
   * 
   * Invariant to 3D rigid head rotations (Yaw, Pitch, Roll) up to +- 30 degrees.
   * Smoothly degenerates to 2D orthogonal projection if z coordinates are zero or absent.
   * 
   * @param landmarks Array of 468 MediaPipe normalized 3D landmarks
   * @returns Normalized asymmetry index (baseline < 0.05, potential paresis >= 0.12)
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

    // Unit lateral axis u_x
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
      // Degenerate geometry: fall back to 2D perpendicular
      vMidX = -uEyesY;
      vMidY = uEyesX;
      vMidZ = 0.0;
    } else {
      vMidX /= normMid;
      vMidY /= normMid;
      vMidZ /= normMid;
    }

    // Ensure u_y points downward toward the chin (positive dot product with vChin)
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
      this.faceMesh.close();
      this.faceMesh = null;
    }
    if (this.offscreenCanvas) {
      this.offscreenCanvas.width = 0;
      this.offscreenCanvas.height = 0;
      this.offscreenCanvas = null;
      this.offscreenCtx = null;
    }
    this.isInitialized = false;
    this.latestResults = null;
    this.resolveFramePromise = null;
    this.prevNasionX = -1;
    this.prevNasionY = -1;
  }
}
