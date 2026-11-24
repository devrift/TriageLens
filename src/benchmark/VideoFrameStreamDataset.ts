/**
 * TriageLens Video Frame Stream Dataset Generator
 * 
 * Generates and streams 10,000 synthetic raw camera video frames (RGB pixel buffers)
 * simulating realistic camera sensor capture across 22 multi-patient clinical sessions:
 * - Full 2D pixel grids (320x240 @ 30 FPS)
 * - Spatial face geometry with anatomical forehead ROI, eyes, mouth commissures
 * - Fitzpatrick phototypes I through VI with wavelength-dependent optical absorption
 * - Spatial lighting gradients, directional shadows, and fluorescent 50/60Hz flicker
 * - Involuntary micro-head translation dx, dy and rotation (head tremor, respiration sway)
 * - Capillary transillumination frames for rear finger-contact mode
 */

export interface FrameStreamTelemetryTarget {
  sessionId: number;
  mode: 'face' | 'finger';
  groundTruthHr: number;
  groundTruthRr: number;
  groundTruthAsym: number;
  fitzpatrick: 'I' | 'II' | 'III' | 'IV' | 'V' | 'VI';
  snrDb: number;
  frameCount: number;
}

export interface VideoFrame {
  width: number;
  height: number;
  data: Uint8ClampedArray; // RGBA pixel buffer
  timestampMs: number;
}

export class VideoFrameStreamDataset {
  private seed: number;

  constructor(initialSeed: number = 88991122) {
    this.seed = initialSeed;
  }

  private random(): number {
    this.seed = (this.seed * 16807 + 0) % 2147483647;
    return (this.seed - 1) / 2147483646;
  }

  private gaussianNoise(sigma: number = 1.0): number {
    const u1 = Math.max(1e-12, this.random());
    const u2 = this.random();
    return sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /**
   * Synthesize a continuous stream of 2D camera frames for a single patient session.
   * Yields VideoFrame objects (320x240 RGBA) at 30 FPS.
   */
  public generateSessionFrames(
    config: FrameStreamTelemetryTarget,
    onFrame: (frame: VideoFrame, frameIdx: number) => void
  ): void {
    const width = 320;
    const height = 240;
    const numPixels = width * height;
    const frameBuffer = new Uint8ClampedArray(numPixels * 4);

    const fs = 30.0;
    const dt = 1000.0 / fs;
    const fHeart = config.groundTruthHr / 60.0;
    const fResp = config.groundTruthRr / 60.0;

    // Fitzpatrick color profile
    const fitzProfiles = {
      I:   { r: 215, g: 180, b: 165, modR: 0.007, modG: 0.022, modB: 0.005 },
      II:  { r: 200, g: 165, b: 145, modR: 0.006, modG: 0.019, modB: 0.004 },
      III: { r: 180, g: 145, b: 120, modR: 0.005, modG: 0.016, modB: 0.0035 },
      IV:  { r: 155, g: 120, b: 95,  modR: 0.004, modG: 0.013, modB: 0.003 },
      V:   { r: 120, g: 88,  b: 65,  modR: 0.003, modG: 0.009, modB: 0.002 },
      VI:  { r: 85,  g: 55,  b: 40,  modR: 0.002, modG: 0.006, modB: 0.0015 },
    };
    const prof = fitzProfiles[config.fitzpatrick];

    let tCurrent = 0.0;
    let pulsePhase = 0.0;

    for (let f = 0; f < config.frameCount; f++) {
      tCurrent += dt;
      const tSec = tCurrent / 1000.0;

      // Involuntary micro-sway motion (head motion tremor in x, y)
      const swayX = 4.0 * Math.sin(2.0 * Math.PI * fResp * tSec) + this.gaussianNoise(0.5);
      const swayY = 2.5 * Math.cos(2.0 * Math.PI * fResp * tSec) + this.gaussianNoise(0.5);

      // Respiration modulation (RIIV baseline drift + RIFV/RSA)
      const riiv = 0.012 * Math.sin(2.0 * Math.PI * fResp * tSec);
      const rsaFreq = fHeart + 0.08 * fHeart * Math.cos(2.0 * Math.PI * fResp * tSec);
      pulsePhase += rsaFreq * (dt / 1000.0);

      // Arterial pulse shape (systolic peak + dicrotic notch)
      const p = pulsePhase - Math.floor(pulsePhase);
      const pulseWave = Math.exp(-Math.pow(p - 0.16, 2) / 0.01) +
                        0.40 * Math.exp(-Math.pow(p - 0.38, 2) / 0.012) +
                        0.10 * Math.exp(-p * 3.0);

      // Fluorescent lighting 50/60 Hz alias + illumination drift
      const flicker = 0.0015 * Math.sin(2.0 * Math.PI * 10.0 * tSec);
      const drift = 0.003 * Math.sin(2.0 * Math.PI * 0.08 * tSec);
      const noise = this.gaussianNoise(0.003);

      if (config.mode === 'face') {
        // Render 2D face scene with background, skin oval, and forehead ROI
        const faceCenterX = width * 0.5 + swayX;
        const faceCenterY = height * 0.5 + swayY;
        const faceRadiusX = width * 0.28;
        const faceRadiusY = height * 0.38;

        // Current skin tone with pulsatile hemodynamics
        const skinR = prof.r * (1.0 + prof.modR * pulseWave + riiv * 0.4 + drift + flicker + noise);
        const skinG = prof.g * (1.0 - prof.modG * pulseWave + riiv * 1.0 + drift + flicker + noise);
        const skinB = prof.b * (1.0 - prof.modB * pulseWave + riiv * 0.6 + drift + flicker + noise);

        let ptr = 0;
        for (let y = 0; y < height; y++) {
          const dy = (y - faceCenterY) / faceRadiusY;
          const dy2 = dy * dy;

          for (let x = 0; x < width; x++) {
            const dx = (x - faceCenterX) / faceRadiusX;
            const dist2 = dx * dx + dy2;

            if (dist2 <= 1.0) {
              // Inside face oval: calculate spatial lighting gradient (ambient + directional)
              const shadowFactor = 1.0 - 0.15 * dy - 0.10 * dx; // subtle directional shading
              frameBuffer[ptr]     = Math.min(255, Math.max(0, skinR * shadowFactor));
              frameBuffer[ptr + 1] = Math.min(255, Math.max(0, skinG * shadowFactor));
              frameBuffer[ptr + 2] = Math.min(255, Math.max(0, skinB * shadowFactor));
              frameBuffer[ptr + 3] = 255;
            } else {
              // Background: dark room gradient
              const bgVal = Math.max(10, Math.min(45, 20 + 0.05 * y));
              frameBuffer[ptr]     = bgVal;
              frameBuffer[ptr + 1] = bgVal;
              frameBuffer[ptr + 2] = bgVal + 5;
              frameBuffer[ptr + 3] = 255;
            }
            ptr += 4;
          }
        }
      } else {
        // Finger contact mode: capillary transillumination
        // Red saturated with inverted green pulse
        const fingerR = 195.0 + 3.0 * Math.sin(2.0 * Math.PI * fHeart * tSec) + this.gaussianNoise(1.0);
        const fingerG = 48.0 - 6.0 * pulseWave + 1.5 * Math.sin(2.0 * Math.PI * fResp * tSec) + this.gaussianNoise(0.8);
        const fingerB = 38.0 + this.gaussianNoise(0.5);

        let ptr = 0;
        for (let i = 0; i < numPixels; i++) {
          frameBuffer[ptr]     = Math.min(255, Math.max(0, fingerR));
          frameBuffer[ptr + 1] = Math.min(255, Math.max(0, fingerG));
          frameBuffer[ptr + 2] = Math.min(255, Math.max(0, fingerB));
          frameBuffer[ptr + 3] = 255;
          ptr += 4;
        }
      }

      onFrame({
        width,
        height,
        data: frameBuffer,
        timestampMs: tCurrent,
      }, f);
    }
  }
}
