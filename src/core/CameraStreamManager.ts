/**
 * TriageLens CameraStreamManager
 * 
 * Hardware video stream lifecycle, mobile camera constraints, and torch control.
 * Strictly typed error handling for permissions and device availability.
 * Race-safe async stream acquisition and mode switching.
 */

import { CameraStreamError } from '../types/triage';

export interface CameraDiagnostics {
  active: boolean;
  facingMode: 'user' | 'environment' | null;
  width: number;
  height: number;
  frameRate: number;
  torchSupported: boolean;
  torchOn: boolean;
}

export class CameraStreamManager {
  private currentStream: MediaStream | null = null;
  private attachedVideoEl: HTMLVideoElement | null = null;
  private isTorchActive: boolean = false;
  private torchSupported: boolean = false;
  private activeRequestId: number = 0;
  private inFlightAcquire: Promise<MediaStream> | null = null;
  private activeFacingMode: 'user' | 'environment' | null = null;
  private activeResolution: { width: number; height: number } | null = null;

  /**
   * Start front-facing camera stream optimized for facial rPPG telemetry.
   */
  public async startFaceStream(
    videoEl: HTMLVideoElement,
    resolution: { width: number; height: number } = { width: 640, height: 480 }
  ): Promise<MediaStream> {
    if (
      this.currentStream &&
      this.currentStream.active &&
      this.activeFacingMode === 'user' &&
      this.attachedVideoEl === videoEl &&
      this.activeResolution?.width === resolution.width &&
      this.activeResolution?.height === resolution.height
    ) {
      return this.currentStream;
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        frameRate: { ideal: 30 },
      },
    };

    return this.acquireStream(videoEl, constraints, false, 'user', resolution);
  }

  /**
   * Start rear-facing camera stream optimized for finger-contact PPG transillumination.
   */
  public async startFingerStream(
    videoEl: HTMLVideoElement,
    resolution: { width: number; height: number } = { width: 320, height: 240 }
  ): Promise<MediaStream> {
    if (
      this.currentStream &&
      this.currentStream.active &&
      this.activeFacingMode === 'environment' &&
      this.attachedVideoEl === videoEl &&
      this.activeResolution?.width === resolution.width &&
      this.activeResolution?.height === resolution.height
    ) {
      return this.currentStream;
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: 'environment',
        width: { ideal: resolution.width },
        height: { ideal: resolution.height },
        frameRate: { ideal: 30 },
      },
    };

    return this.acquireStream(videoEl, constraints, true, 'environment', resolution);
  }

  /**
   * Acquire stream via getUserMedia and attach to video element in a race-safe manner.
   */
  private async acquireStream(
    videoEl: HTMLVideoElement,
    constraints: MediaStreamConstraints,
    attemptTorch: boolean,
    facingMode: 'user' | 'environment',
    resolution: { width: number; height: number }
  ): Promise<MediaStream> {
    // If an acquisition with matching parameters is already in flight, await it
    if (
      this.inFlightAcquire &&
      this.activeFacingMode === facingMode &&
      this.activeResolution?.width === resolution.width
    ) {
      return this.inFlightAcquire;
    }

    const requestId = ++this.activeRequestId;
    this.activeFacingMode = facingMode;
    this.activeResolution = resolution;

    const acquirePromise: Promise<MediaStream> = (async (): Promise<MediaStream> => {
      // Release previous stream WITHOUT bumping activeRequestId so we don't invalidate ourselves
      this.releaseActiveStream(videoEl);

      if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new CameraStreamError('CAMERA_NOT_FOUND', 'MediaDevices API is not supported in this runtime environment.');
      }

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err: any) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          throw new CameraStreamError('CAMERA_PERMISSION_DENIED', 'Camera access was denied by the user or security policy.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          throw new CameraStreamError('CAMERA_NOT_FOUND', 'Requested camera device was not found on this hardware.');
        }
        throw new CameraStreamError('CAMERA_ERROR', err.message || 'Failed to acquire camera stream.');
      }

      // Race condition check: if a newer request was issued while waiting for getUserMedia,
      // silently discard this obsolete stream without bubbling an error to UI.
      if (requestId !== this.activeRequestId) {
        stream.getTracks().forEach((t) => t.stop());
        if (this.currentStream && this.currentStream.active) {
          return this.currentStream;
        }
        return stream;
      }

      const videoTrack = stream.getVideoTracks()[0];
      if (!videoTrack) {
        throw new CameraStreamError('CAMERA_NOT_FOUND', 'MediaStream contains no video tracks.');
      }

      const caps = typeof videoTrack.getCapabilities === 'function'
        ? (videoTrack.getCapabilities() as any)
        : null;
      this.torchSupported = Boolean(caps && caps.torch);

      if (attemptTorch) {
        await this.applyTorchSafely(videoTrack);
      }

      this.currentStream = stream;
      this.attachedVideoEl = videoEl;

      videoEl.srcObject = stream;
      videoEl.playsInline = true;
      videoEl.muted = true;
      videoEl.autoplay = true;

      try {
        await videoEl.play();
      } catch {
        // Autoplay can be subject to user gesture policies in some browsers
      }

      return stream;
    })();

    this.inFlightAcquire = acquirePromise;
    try {
      return await acquirePromise;
    } finally {
      if (this.inFlightAcquire === acquirePromise) {
        this.inFlightAcquire = null;
      }
    }
  }

  /**
   * Inspect track capabilities and engage torch if supported, gracefully handling errors.
   */
  public async applyTorchSafely(videoTrack: MediaStreamTrack, enable: boolean = true): Promise<boolean> {
    const capabilities = typeof videoTrack.getCapabilities === 'function'
      ? (videoTrack.getCapabilities() as any)
      : null;

    this.torchSupported = Boolean(capabilities && capabilities.torch);
    if (!this.torchSupported) {
      this.isTorchActive = false;
      return false;
    }

    try {
      await videoTrack.applyConstraints({
        advanced: [{ torch: enable } as any],
      });
      this.isTorchActive = enable;
      return true;
    } catch {
      this.isTorchActive = false;
      return false;
    }
  }

  public async setTorch(enable: boolean): Promise<boolean> {
    if (!this.currentStream) return false;
    const track = this.currentStream.getVideoTracks()[0];
    if (!track) return false;
    return this.applyTorchSafely(track, enable);
  }

  /**
   * Stop active camera stream, invalidate pending acquisitions, release tracks, and detach video element.
   */
  public stopStream(videoEl?: HTMLVideoElement): void {
    this.activeRequestId++;
    this.releaseActiveStream(videoEl);
  }

  private releaseActiveStream(videoEl?: HTMLVideoElement): void {
    if (this.currentStream) {
      const tracks = this.currentStream.getTracks();
      for (let i = 0; i < tracks.length; i++) {
        const track = tracks[i];
        if (this.isTorchActive && typeof track.applyConstraints === 'function') {
          try {
            track.applyConstraints({ advanced: [{ torch: false } as any] }).catch(() => {});
          } catch {}
        }
        track.stop();
      }
      this.currentStream = null;
    }

    if (videoEl) {
      videoEl.srcObject = null;
    }

    if (this.attachedVideoEl) {
      this.attachedVideoEl.srcObject = null;
      this.attachedVideoEl = null;
    }

    this.isTorchActive = false;
  }

  public getDiagnostics(): CameraDiagnostics {
    const track = this.currentStream?.getVideoTracks()[0];
    const settings = track && typeof track.getSettings === 'function' ? track.getSettings() : null;
    return {
      active: Boolean(this.currentStream && this.currentStream.active),
      facingMode: this.activeFacingMode,
      width: settings?.width ?? this.activeResolution?.width ?? 0,
      height: settings?.height ?? this.activeResolution?.height ?? 0,
      frameRate: Math.round(settings?.frameRate ?? 0),
      torchSupported: this.torchSupported,
      torchOn: this.isTorchActive,
    };
  }

  public getStream(): MediaStream | null {
    return this.currentStream;
  }

  public isTorchOn(): boolean {
    return this.isTorchActive;
  }

  public isTorchSupported(): boolean {
    return this.torchSupported;
  }
}
