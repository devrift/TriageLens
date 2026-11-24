/**
 * TriageLens CameraStreamManager
 * 
 * Hardware video stream lifecycle, mobile camera constraints, and torch control.
 * Strictly typed error handling for permissions and device availability.
 * Race-safe async stream acquisition and mode switching.
 */

import { CameraStreamError } from '../types/triage';

export class CameraStreamManager {
  private currentStream: MediaStream | null = null;
  private attachedVideoEl: HTMLVideoElement | null = null;
  private isTorchActive: boolean = false;
  private activeRequestId: number = 0;
  private inFlightAcquire: Promise<MediaStream> | null = null;
  private activeFacingMode: 'user' | 'environment' | null = null;

  /**
   * Start front-facing camera stream optimized for facial rPPG telemetry:
   * 640x480 @ 30 FPS ideal constraints.
   */
  public async startFaceStream(videoEl: HTMLVideoElement): Promise<MediaStream> {
    if (this.currentStream && this.currentStream.active && this.activeFacingMode === 'user' && this.attachedVideoEl === videoEl) {
      return this.currentStream;
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
      },
    };

    return this.acquireStream(videoEl, constraints, false, 'user');
  }

  /**
   * Start rear-facing camera stream optimized for finger-contact PPG transillumination:
   * 320x240 @ 30 FPS ideal constraints with white LED torch engaged when supported.
   */
  public async startFingerStream(videoEl: HTMLVideoElement): Promise<MediaStream> {
    if (this.currentStream && this.currentStream.active && this.activeFacingMode === 'environment' && this.attachedVideoEl === videoEl) {
      return this.currentStream;
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: 'environment',
        width: { ideal: 320 },
        height: { ideal: 240 },
        frameRate: { ideal: 30 },
      },
    };

    return this.acquireStream(videoEl, constraints, true, 'environment');
  }

  /**
   * Acquire stream via getUserMedia and attach to video element in a race-safe manner.
   */
  private async acquireStream(
    videoEl: HTMLVideoElement,
    constraints: MediaStreamConstraints,
    attemptTorch: boolean,
    facingMode: 'user' | 'environment'
  ): Promise<MediaStream> {
    // If an acquisition with matching parameters is already in flight, await it
    if (this.inFlightAcquire && this.activeFacingMode === facingMode) {
      return this.inFlightAcquire;
    }

    const requestId = ++this.activeRequestId;
    this.activeFacingMode = facingMode;

    const acquirePromise: Promise<MediaStream> = (async (): Promise<MediaStream> => {
      this.stopStream(videoEl);

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
  public async applyTorchSafely(videoTrack: MediaStreamTrack): Promise<boolean> {
    const capabilities = typeof videoTrack.getCapabilities === 'function'
      ? (videoTrack.getCapabilities() as any)
      : null;

    if (!capabilities || !capabilities.torch) {
      this.isTorchActive = false;
      return false;
    }

    try {
      await videoTrack.applyConstraints({
        advanced: [{ torch: true } as any],
      });
      this.isTorchActive = true;
      return true;
    } catch {
      this.isTorchActive = false;
      return false;
    }
  }

  /**
   * Stop active camera stream, release all tracks, detach video element, and reset torch.
   */
  public stopStream(videoEl?: HTMLVideoElement): void {
    // Invalidate any in-flight stream requests
    this.activeRequestId++;

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

  public getStream(): MediaStream | null {
    return this.currentStream;
  }

  public isTorchOn(): boolean {
    return this.isTorchActive;
  }
}
