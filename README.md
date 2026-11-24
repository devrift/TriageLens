# TriageLens

> **Zero-Hardware, Offline-First Mobile Vital Sign Telemetry Engine**  
> Runs 100% locally on device in mobile web runtimes (Android Chrome / iOS Safari PWA) via WebAssembly, WebGL, and digital signal processing (DSP).

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![TypeScript: Strict](https://img.shields.io/badge/TypeScript-Strict-blue.svg)](tsconfig.json)
[![Tests: 43 Passed](https://img.shields.io/badge/Tests-43%20Passed-brightgreen.svg)](src/core/__tests__)
[![PWA: Offline Ready](https://img.shields.io/badge/PWA-Offline%20Ready-black.svg)](public/manifest.webmanifest)

---

## 🌟 Key Capabilities

1. **Contactless Facial rPPG (Remote Photoplethysmography):**
   - Forehead region-of-interest (ROI) extraction clamped safely across varied face shapes.
   - Plane-Orthogonal-to-Skin (**POS**) chrominance projection algorithm.
   - Real-time pulse signal decomposition without pixel or numerical data leaving the device.

2. **FAST Bilateral Facial Stroke Asymmetry Screening:**
   - 468-point 3D facial landmark mesh tracking.
   - Dynamic vertical displacement of left and right oral commissures relative to the vertical midline axis anchored at Nasion (168) and Chin Tip (152), normalized by inter-ocular distance ($d_{\text{IOD}}$).
   - Physiological baseline: $\Delta_{\text{asymmetry}} < 0.05$; alert threshold: $\ge 0.12$.

3. **Contact Finger PPG (Rear Camera + Torch):**
   - Transillumination capillary gate testing deep-red saturation and green-channel absorption ($R > 130, G < 100, R/G > 1.8$).
   - High-sensitivity green channel inversion for pulse wave polarity.

4. **Deterministic Classical Signal Processing (Zero AI Hallucinations for Vitals):**
   - Cascade of two 2nd-order Direct Form II Transposed biquad sections (4th-order Butterworth bandpass configured for $0.75 - 3.0\text{ Hz}$ / $45 - 180\text{ BPM}$ at $30\text{ Hz}$).
   - Adaptive amplitude thresholding ($\mu + 0.4\sigma$) with $\ge 300\text{ ms}$ refractory period lockout ($200\text{ BPM}$ upper ceiling).
   - 256-point Radix-2 FFT Power Spectral Density evaluation for Signal Quality Index (**SQI**).
   - Respiration rate extraction via low-frequency envelope spectral analysis ($0.15 - 0.50\text{ Hz}$ / $9 - 30\text{ RPM}$).

5. **AI Clinical Triage Companion:**
   - Offline clinical rule engine mapping vitals to severity levels (`NORMAL`, `ELEVATED`, `CRITICAL`, `INVALID`).
   - Optional online Gemini or OpenRouter guidance (strictly transmits numerical telemetry, never raw camera frames).

6. **Local Session Persistence:**
   - Stores scan history and baseline delta comparisons locally with automatic in-memory fallback.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/devrift/TriageLens.git
cd TriageLens

# Install dependencies
npm install
```

### Development & Testing

```bash
# Start local development server
npm run dev

# Run strict type checking
npm run typecheck

# Run test suite (43 vitest unit & integration tests)
npm test

# Build for production
npm run build
```

---

## 📱 Standalone APK & PWA

- **Android APK:** Download the standalone release package from the UI or `/triagelens-release.apk`.
- **Progressive Web App (PWA):** Open the site on mobile Safari or Chrome and select **"Add to Home Screen"** for offline biometric telemetry.

---

## 🔒 Privacy & Invariants

- **Zero External Backend for Signal Processing:** 100% of landmark tracking, ROI extraction, filtering, and stroke geometry executes locally on device CPU/GPU.
- **Zero Video Transmission:** Pixels and camera streams never leave the client device.
- **Secret Safety:** `.env*` files are strictly git-ignored.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
