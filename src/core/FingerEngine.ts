/**
 * TriageLens FingerEngine
 * 
 * Rear camera contact photoplethysmography (PPG) sampler.
 * Evaluates capillary transillumination over persistent 32x32 offscreen canvas.
 * Zero-allocation during active sampling.
 */

import { FingerSampleResult } from '../types/triage';

export class FingerEngine {
  public static readonly CANVAS_SIZE: number = 32;

  // Contact gate thresholds: R > 130, G < 100, R / (G + eps) > 1.8
  public static readonly MIN_RED_INTENSITY: number = 130.0;
  public static readonly MAX_GREEN_INTENSITY: number = 100.0;
  public static readonly MIN_RED_TO_GREEN_RATIO: number = 1.8;

  private offscreenCanvas: HTMLCanvasElement | null = null;
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  // Static pre-allocated result object for zero GC pause
  private readonly cachedResult: FingerSampleResult = {
    isFingerPlaced: false,
    signalValue: 0.0,
    rgb: [0, 0, 0],
    luma: 0.0,
    isPlacementValid: false,
    timestamp: 0,
  };

  constructor() {
    if (typeof document !== 'undefined') {
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCanvas.width = FingerEngine.CANVAS_SIZE;
      this.offscreenCanvas.height = FingerEngine.CANVAS_SIZE;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', {
        willReadFrequently: true,
      });
    }
  }

  /**
   * Pure mathematical validator for capillary transillumination gate.
   * Capillary transillumination under white LED light saturates human tissue in deep red
   * while absorbing green.
   */
  public static validateContact(r: number, g: number, _b?: number, luma?: number): boolean {
    const isRedSaturated = r > FingerEngine.MIN_RED_INTENSITY;
    const isGreenAbsorbed = g < FingerEngine.MAX_GREEN_INTENSITY;
    const ratio = r / (g + 1e-6);
    const isRatioValid = ratio > FingerEngine.MIN_RED_TO_GREEN_RATIO;
    const calculatedLuma = luma !== undefined ? luma : (0.299 * r + 0.587 * g + 0.114 * (_b ?? 0));
    const isLumaSane = calculatedLuma > 30.0 && calculatedLuma < 252.0;

    return isRedSaturated && isGreenAbsorbed && isRatioValid && isLumaSane;
  }

  /**
   * Sample the central 32x32 region of the rear camera feed and evaluate placement.
   */
  public sampleFingerFrame(videoEl: HTMLVideoElement): FingerSampleResult {
    const timestamp = typeof performance !== 'undefined' ? performance.now() : Date.now();

    if (!this.offscreenCanvas || !this.offscreenCtx) {
      this.cachedResult.rgb[0] = 0;
      this.cachedResult.rgb[1] = 0;
      this.cachedResult.rgb[2] = 0;
      this.cachedResult.luma = 0;
      this.cachedResult.isFingerPlaced = false;
      this.cachedResult.isPlacementValid = false;
      this.cachedResult.signalValue = 0;
      this.cachedResult.timestamp = timestamp;
      return this.cachedResult;
    }

    const vw = videoEl.videoWidth || 320;
    const vh = videoEl.videoHeight || 240;

    // Crop central square region from video
    const cropSize = Math.floor(Math.min(vw, vh) * 0.4);
    const sx = Math.floor((vw - cropSize) / 2);
    const sy = Math.floor((vh - cropSize) / 2);

    // Draw central sub-region scaled down to 32x32
    this.offscreenCtx.drawImage(
      videoEl,
      sx,
      sy,
      cropSize,
      cropSize,
      0,
      0,
      FingerEngine.CANVAS_SIZE,
      FingerEngine.CANVAS_SIZE
    );

    const imgData = this.offscreenCtx.getImageData(
      0,
      0,
      FingerEngine.CANVAS_SIZE,
      FingerEngine.CANVAS_SIZE
    );
    const data = imgData.data;
    const totalPixels = FingerEngine.CANVAS_SIZE * FingerEngine.CANVAS_SIZE; // 1024

    let sumR = 0;
    let sumG = 0;
    let sumB = 0;

    for (let i = 0; i < data.length; i += 4) {
      sumR += data[i];
      sumG += data[i + 1];
      sumB += data[i + 2];
    }

    const meanR = sumR / totalPixels;
    const meanG = sumG / totalPixels;
    const meanB = sumB / totalPixels;
    const luma = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;

    const isPlacementValid = FingerEngine.validateContact(meanR, meanG, meanB, luma);
    // Green channel inverted and normalized for pulse wave polarity
    const signalValue = -meanG / (meanR + 1e-4);

    this.cachedResult.rgb[0] = meanR;
    this.cachedResult.rgb[1] = meanG;
    this.cachedResult.rgb[2] = meanB;
    this.cachedResult.luma = luma;
    this.cachedResult.isFingerPlaced = isPlacementValid;
    this.cachedResult.isPlacementValid = isPlacementValid;
    this.cachedResult.signalValue = signalValue;
    this.cachedResult.timestamp = timestamp;

    return this.cachedResult;
  }

  /**
   * Release resources
   */
  public destroy(): void {
    if (this.offscreenCanvas) {
      this.offscreenCanvas.width = 0;
      this.offscreenCanvas.height = 0;
      this.offscreenCanvas = null;
      this.offscreenCtx = null;
    }
  }
}
