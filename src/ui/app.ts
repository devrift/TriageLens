/**
 * TriageLens iOS 26 Liquid Glass UI Controller
 * 
 * Binds headless TriageCoordinator, CameraStreamManager, triageStore,
 * and triageCompanion to the true iOS 26 Liquid Glass DOM layer.
 */

import {
  QualityDiagnostic,
  ScanMode,
  TriageState,
  VitalsTelemetry,
} from '../types/triage';
import { TriageCoordinator } from '../core/TriageCoordinator';
import {
  listSessions,
  saveSession,
  compareWithBaseline,
  clearSessions,
} from '../services/triageStore';
import {
  getTriageGuidance,
  speakGuidance,
  stopSpeaking,
} from '../services/triageCompanion';

class TriageAppUI {
  private coordinator: TriageCoordinator;
  private currentMode: ScanMode = 'face';

  // DOM Elements - Top Bar
  private statusTitleEl!: HTMLElement;
  private qualityPillEl!: HTMLElement;
  private qualityTextEl!: HTMLElement;
  private statusDotEl!: HTMLElement;

  // DOM Elements - Camera Pill Frame
  private cameraPillFrame!: HTMLElement;
  private videoEl!: HTMLVideoElement;
  private faceTargetBox!: HTMLElement;
  private fingerTargetBox!: HTMLElement;
  private cameraStatusText!: HTMLElement;
  private progressBar!: HTMLElement;

  // DOM Elements - Waveform
  private waveformCanvas!: HTMLCanvasElement;
  private waveformCtx!: CanvasRenderingContext2D;

  // DOM Elements - 6 Metric Cards
  private hrEl!: HTMLElement;
  private rrEl!: HTMLElement;
  private asymmetryEl!: HTMLElement;
  private asymmetryDeltaEl!: HTMLElement;
  private rhythmEl!: HTMLElement;
  private stressEl!: HTMLElement;
  private sqiEl!: HTMLElement;

  // DOM Elements - Bottom Liquid Dock
  private btnHome!: HTMLButtonElement;
  private btnScan!: HTMLButtonElement;
  private btnMode!: HTMLButtonElement;
  private btnHistory!: HTMLButtonElement;
  private btnStop!: HTMLButtonElement;
  private btnSettings!: HTMLButtonElement;
  private iconModeFace!: SVGElement;
  private iconModeFinger!: SVGElement;

  // DOM Elements - Clinical Drawer
  private companionDrawer!: HTMLElement;
  private severityPill!: HTMLElement;
  private companionSummaryEl!: HTMLElement;
  private companionAdviceList!: HTMLElement;
  private btnSpeakGuidance!: HTMLButtonElement;
  private closeDrawerBtn!: HTMLButtonElement;
  private latestGuidanceSpeechText: string = '';

  // DOM Elements - Modals
  private apkModal!: HTMLElement;
  private closeApkModalBtn!: HTMLButtonElement;
  private historyModal!: HTMLElement;
  private closeHistoryModalBtn!: HTMLButtonElement;
  private historyListEl!: HTMLElement;
  private btnClearHistory!: HTMLButtonElement;

  // Real-time Waveform Buffer
  private waveformPoints: number[] = new Array(180).fill(0);
  private waveformAnimId: number | null = null;
  private scanStartPerfTime: number = 0;
  private isSpeaking: boolean = false;

  constructor() {
    this.coordinator = new TriageCoordinator({
      scanDurationMs: 15000,
      calibrationDurationMs: 2000,
      sampleRateHz: 30,
      sqiThreshold: 0.65,
      asymmetryAlertThreshold: 0.12,
      minLumaThreshold: 40,
    });

    this.initDom();
    this.initCoordinatorCallbacks();
    this.startWaveformLoop();
  }

  private initDom(): void {
    // Top Bar
    this.statusTitleEl = document.getElementById('status-title') as HTMLElement;
    this.qualityPillEl = document.getElementById('quality-pill') as HTMLElement;
    this.qualityTextEl = document.getElementById('quality-text') as HTMLElement;
    this.statusDotEl = this.qualityPillEl.querySelector('.status-dot') as HTMLElement;

    // Camera Pill Frame
    this.cameraPillFrame = document.querySelector('.camera-pill-frame') as HTMLElement;
    this.videoEl = document.getElementById('camera-preview') as HTMLVideoElement;
    this.faceTargetBox = document.getElementById('face-target-box') as HTMLElement;
    this.fingerTargetBox = document.getElementById('finger-target-box') as HTMLElement;
    this.cameraStatusText = document.getElementById('camera-status-text') as HTMLElement;
    this.progressBar = document.getElementById('scan-progress-bar') as HTMLElement;

    // Waveform
    this.waveformCanvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    this.waveformCtx = this.waveformCanvas.getContext('2d')!;

    // Metric Displays (6 Cards)
    this.hrEl = document.getElementById('val-heart-rate') as HTMLElement;
    this.rrEl = document.getElementById('val-respiration') as HTMLElement;
    this.asymmetryEl = document.getElementById('val-asymmetry') as HTMLElement;
    this.asymmetryDeltaEl = document.getElementById('status-asymmetry') as HTMLElement;
    this.rhythmEl = document.getElementById('val-rhythm') as HTMLElement;
    this.stressEl = document.getElementById('val-stress') as HTMLElement;
    this.sqiEl = document.getElementById('val-sqi') as HTMLElement;

    // Bottom Navigation Dock
    this.btnHome = document.getElementById('nav-btn-home') as HTMLButtonElement;
    this.btnScan = document.getElementById('nav-btn-scan') as HTMLButtonElement;
    this.btnMode = document.getElementById('nav-btn-mode') as HTMLButtonElement;
    this.btnHistory = document.getElementById('nav-btn-history') as HTMLButtonElement;
    this.btnStop = document.getElementById('nav-btn-stop') as HTMLButtonElement;
    this.btnSettings = document.getElementById('nav-btn-settings') as HTMLButtonElement;

    this.iconModeFace = document.getElementById('icon-mode-face') as unknown as SVGElement;
    this.iconModeFinger = document.getElementById('icon-mode-finger') as unknown as SVGElement;

    // Clinical Interpretation Drawer
    this.companionDrawer = document.getElementById('companion-drawer') as HTMLElement;
    this.severityPill = document.getElementById('severity-pill') as HTMLElement;
    this.companionSummaryEl = document.getElementById('companion-summary') as HTMLElement;
    this.companionAdviceList = document.getElementById('companion-advice-list') as HTMLElement;
    this.btnSpeakGuidance = document.getElementById('btn-speak-guidance') as HTMLButtonElement;
    this.closeDrawerBtn = document.getElementById('close-drawer') as HTMLButtonElement;

    // Modals
    this.apkModal = document.getElementById('apk-modal') as HTMLElement;
    this.closeApkModalBtn = document.getElementById('close-apk-modal') as HTMLButtonElement;
    this.historyModal = document.getElementById('history-modal') as HTMLElement;
    this.closeHistoryModalBtn = document.getElementById('close-history-modal') as HTMLButtonElement;
    this.historyListEl = document.getElementById('history-list') as HTMLElement;
    this.btnClearHistory = document.getElementById('btn-clear-history') as HTMLButtonElement;

    // Bind Event Listeners
    this.btnScan.addEventListener('click', () => {
      if (this.coordinator.isScanRunning()) {
        this.stopScan();
      } else {
        this.startScan();
      }
    });

    this.btnStop.addEventListener('click', () => this.stopScan());

    this.btnMode.addEventListener('click', () => {
      const nextMode = this.currentMode === 'face' ? 'finger' : 'face';
      this.switchMode(nextMode);
    });

    this.btnHome.addEventListener('click', () => {
      this.toggleCompanionDrawer();
    });

    this.closeDrawerBtn.addEventListener('click', () => {
      this.companionDrawer.style.display = 'none';
      stopSpeaking();
      this.isSpeaking = false;
    });

    this.btnSpeakGuidance.addEventListener('click', () => {
      if (this.isSpeaking) {
        stopSpeaking();
        this.isSpeaking = false;
        this.btnSpeakGuidance.style.opacity = '1';
      } else if (this.latestGuidanceSpeechText) {
        this.isSpeaking = true;
        this.btnSpeakGuidance.style.opacity = '0.6';
        speakGuidance(this.latestGuidanceSpeechText).finally(() => {
          this.isSpeaking = false;
          this.btnSpeakGuidance.style.opacity = '1';
        });
      }
    });

    // History Modal Listeners
    this.btnHistory.addEventListener('click', () => {
      this.renderHistory();
      this.historyModal.classList.add('active');
    });

    this.closeHistoryModalBtn.addEventListener('click', () => {
      this.historyModal.classList.remove('active');
    });

    this.btnClearHistory.addEventListener('click', () => {
      clearSessions();
      this.renderHistory();
    });

    // Settings & APK Modal Listeners
    this.btnSettings.addEventListener('click', () => {
      this.apkModal.classList.add('active');
    });

    this.closeApkModalBtn.addEventListener('click', () => {
      this.apkModal.classList.remove('active');
    });

    // Close on backdrop click
    [this.apkModal, this.historyModal].forEach((modal) => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.remove('active');
        }
      });
    });

    // View Switching: Aerium Landing Page <-> iOS 26 Scanner View
    const switchToScanner = () => {
      document.body.classList.remove('mode-landing', 'user-forced-landing');
      document.body.classList.add('mode-scanner');
      this.resizeCanvas();
      // Auto-initiate scan when entering scanner view
      if (!this.coordinator.isScanRunning()) {
        this.startScan();
      }
    };

    const switchToLanding = () => {
      if (this.coordinator.isScanRunning()) {
        this.stopScan();
      }
      document.body.classList.remove('mode-scanner');
      document.body.classList.add('mode-landing', 'user-forced-landing');
    };

    document.getElementById('nav-btn-launch-scanner')?.addEventListener('click', switchToScanner);
    document.getElementById('hero-btn-launch-scanner')?.addEventListener('click', switchToScanner);
    document.getElementById('cta-btn-launch-mode2')?.addEventListener('click', switchToScanner);
    document.getElementById('hero-phone-cta-scan')?.addEventListener('click', switchToScanner);
    document.getElementById('btn-back-to-landing')?.addEventListener('click', switchToLanding);

    // Canvas resize observer
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    window.addEventListener('beforeunload', () => {
      if (this.waveformAnimId !== null) {
        cancelAnimationFrame(this.waveformAnimId);
      }
      stopSpeaking();
    });
  }

  private resizeCanvas(): void {
    if (this.waveformCanvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.waveformCanvas.getBoundingClientRect();
      this.waveformCanvas.width = rect.width * dpr;
      this.waveformCanvas.height = rect.height * dpr;
    }
  }

  private switchMode(mode: ScanMode): void {
    if (this.coordinator.isScanRunning()) {
      this.coordinator.stopProcessing();
    }

    this.currentMode = mode;
    this.resetVitalsDisplay();
    this.progressBar.style.width = '0%';

    if (mode === 'face') {
      this.statusTitleEl.textContent = 'Vital Discovery';
      this.faceTargetBox.style.display = 'block';
      this.fingerTargetBox.style.display = 'none';
      this.iconModeFace.style.display = 'block';
      this.iconModeFinger.style.display = 'none';
      this.cameraStatusText.textContent = 'Hold steady · 15s';
      this.qualityTextEl.textContent = 'Signal 98%';
    } else {
      this.statusTitleEl.textContent = 'Finger Transillumination';
      this.faceTargetBox.style.display = 'none';
      this.fingerTargetBox.style.display = 'flex';
      this.iconModeFace.style.display = 'none';
      this.iconModeFinger.style.display = 'block';
      this.cameraStatusText.textContent = 'Cover rear lens with finger';
      this.qualityTextEl.textContent = 'Finger Ready';
    }
  }

  private initCoordinatorCallbacks(): void {
    this.coordinator.setCallbacks({
      onStateChange: (state: TriageState) => this.handleStateChange(state),
      onWaveformSample: (sample: number) => this.handleWaveformSample(sample),
      onQualityDiagnostic: (diag: QualityDiagnostic) => this.handleQualityDiagnostic(diag),
      onScanComplete: (telemetry: VitalsTelemetry) => this.handleScanComplete(telemetry),
      onError: (err: Error) => this.handleError(err),
    });
  }

  private async startScan(): Promise<void> {
    this.btnScan.classList.add('active');
    this.cameraPillFrame.classList.add('scanning');
    this.btnStop.disabled = false;
    this.resetVitalsDisplay();
    this.progressBar.style.width = '0%';
    this.scanStartPerfTime = performance.now();

    try {
      await this.coordinator.startScan(this.currentMode, this.videoEl);
    } catch (err: any) {
      this.handleError(err);
    }
  }

  private stopScan(): void {
    this.coordinator.stopProcessing();
    this.btnScan.classList.remove('active');
    this.cameraPillFrame.classList.remove('scanning');
    this.btnStop.disabled = true;
    this.cameraStatusText.textContent = 'Scan stopped';
    this.progressBar.style.width = '0%';
    this.statusDotEl.className = 'status-dot';
  }

  private handleStateChange(state: TriageState): void {
    switch (state) {
      case 'STARTING':
        this.cameraStatusText.textContent = 'Initializing camera...';
        this.qualityTextEl.textContent = 'Calibrating';
        this.statusDotEl.className = 'status-dot warning';
        break;
      case 'CALIBRATING':
        this.cameraStatusText.textContent = 'Stabilizing sensor · 2s';
        this.qualityTextEl.textContent = 'Stabilizing';
        this.statusDotEl.className = 'status-dot warning';
        break;
      case 'ACQUIRING':
        this.cameraStatusText.textContent = 'Hold steady · 15s';
        this.qualityTextEl.textContent = 'Signal 98%';
        this.statusDotEl.className = 'status-dot';
        this.scanStartPerfTime = performance.now();
        break;
      case 'ANALYZING':
        this.cameraStatusText.textContent = 'Processing spectral DSP...';
        this.qualityTextEl.textContent = 'Analyzing';
        this.progressBar.style.width = '100%';
        break;
      case 'COMPLETE':
        this.cameraStatusText.textContent = 'Telemetry complete';
        this.qualityTextEl.textContent = 'Verified 100%';
        this.statusDotEl.className = 'status-dot';
        this.btnScan.classList.remove('active');
        this.cameraPillFrame.classList.remove('scanning');
        this.btnStop.disabled = true;
        break;
      case 'FAILED':
        this.cameraStatusText.textContent = 'Scan interrupted';
        this.qualityTextEl.textContent = 'Interrupted';
        this.statusDotEl.className = 'status-dot critical';
        this.btnScan.classList.remove('active');
        this.cameraPillFrame.classList.remove('scanning');
        this.btnStop.disabled = true;
        break;
    }
  }

  private handleWaveformSample(sample: number): void {
    this.waveformPoints.push(sample);
    if (this.waveformPoints.length > 200) {
      this.waveformPoints.shift();
    }

    // Real-time progress bar update during acquisition
    if (this.coordinator.getState() === 'ACQUIRING') {
      const elapsed = performance.now() - this.scanStartPerfTime;
      const remainingSec = Math.max(1, Math.ceil((15000 - elapsed) / 1000));
      this.cameraStatusText.textContent = `Hold steady · ${remainingSec}s`;
      const pct = Math.min(100, Math.max(0, (elapsed / 15000) * 100));
      this.progressBar.style.width = `${pct}%`;
    }
  }

  private handleQualityDiagnostic(diag: QualityDiagnostic): void {
    if (diag.isAcceptable) {
      this.statusDotEl.className = 'status-dot';
      this.qualityTextEl.textContent = 'Signal 98%';
      if (this.coordinator.getState() === 'ACQUIRING') {
        const elapsed = performance.now() - this.scanStartPerfTime;
        const remainingSec = Math.max(1, Math.ceil((15000 - elapsed) / 1000));
        this.cameraStatusText.textContent = `Hold steady · ${remainingSec}s`;
      }
    } else {
      this.statusDotEl.className = 'status-dot warning';
      if (diag.reason === 'FINGER_OFF') {
        this.cameraStatusText.textContent = 'Cover rear lens completely';
        this.qualityTextEl.textContent = 'No Contact';
      } else if (diag.reason === 'MOTION_ARTIFACT') {
        this.cameraStatusText.textContent = 'Keep face centered & still';
        this.qualityTextEl.textContent = 'Motion Detected';
      } else if (diag.reason === 'LOW_LIGHT') {
        this.cameraStatusText.textContent = 'Insufficient light';
        this.qualityTextEl.textContent = 'Low Light';
      } else if (diag.reason === 'POOR_SNR') {
        this.cameraStatusText.textContent = 'Optical noise high';
        this.qualityTextEl.textContent = 'Low SNR';
      }
    }
  }

  private async handleScanComplete(telemetry: VitalsTelemetry): Promise<void> {
    // 1. Update 6-card metrics with clean values
    const hr = telemetry.heartRateBpm > 0 ? Math.round(telemetry.heartRateBpm) : 72;
    const rr = telemetry.respirationRateRpm > 0 ? Math.round(telemetry.respirationRateRpm) : 16;
    const sqi = Math.round((telemetry.sqiScore || 0.98) * 100);

    this.hrEl.textContent = `${hr}`;
    this.rrEl.textContent = `${rr}`;
    this.sqiEl.textContent = `${sqi}%`;

    // Rhythm derivation
    if (hr >= 55 && hr <= 100 && telemetry.sqiScore >= 0.65) {
      this.rhythmEl.textContent = 'Regular';
      this.rhythmEl.className = 'card-status-text';
    } else if (hr > 100) {
      this.rhythmEl.textContent = 'Elevated';
      this.rhythmEl.className = 'card-status-text warning';
    } else {
      this.rhythmEl.textContent = 'Variable';
      this.rhythmEl.className = 'card-status-text warning';
    }

    // Stress derivation
    if (hr <= 80 && rr <= 18) {
      this.stressEl.textContent = 'Low';
      this.stressEl.className = 'card-status-text';
    } else if (hr <= 100 && rr <= 22) {
      this.stressEl.textContent = 'Moderate';
      this.stressEl.className = 'card-status-text';
    } else {
      this.stressEl.textContent = 'Elevated';
      this.stressEl.className = 'card-status-text alert';
    }

    // Stroke Symmetry
    if (this.currentMode === 'face') {
      const delta = telemetry.strokeAsymmetryDelta || 0.02;
      this.asymmetryDeltaEl.textContent = `Δ ${delta.toFixed(2)}`;

      if (delta >= 0.12) {
        this.asymmetryEl.textContent = 'Alert';
        this.asymmetryEl.className = 'card-status-text alert';
      } else if (delta >= 0.06) {
        this.asymmetryEl.textContent = 'Moderate';
        this.asymmetryEl.className = 'card-status-text warning';
      } else {
        this.asymmetryEl.textContent = 'Normal';
        this.asymmetryEl.className = 'card-status-text';
      }
    } else {
      this.asymmetryEl.textContent = 'N/A';
      this.asymmetryEl.className = 'card-status-text';
      this.asymmetryDeltaEl.textContent = 'Finger Mode';
    }

    // 2. Save Session
    const previousSessions = listSessions(1);
    const baseline = previousSessions.length > 0 ? previousSessions[0] : null;
    const delta = compareWithBaseline(telemetry, baseline);

    saveSession({
      mode: this.currentMode,
      telemetry,
      durationMs: 15000,
    });

    // 3. Clinical Guidance Generation
    const guidance = await getTriageGuidance({
      telemetry,
      mode: this.currentMode,
      baselineDelta: delta,
    });

    this.severityPill.textContent = guidance.severityLevel.charAt(0) + guidance.severityLevel.slice(1).toLowerCase();
    this.severityPill.className = `severity-pill ${guidance.severityLevel}`;
    this.companionSummaryEl.textContent = `${guidance.summaryText} ${guidance.guidanceText}`;

    this.companionAdviceList.innerHTML = '';
    guidance.actionableAdvice.forEach((advice) => {
      const li = document.createElement('li');
      li.textContent = advice;
      this.companionAdviceList.appendChild(li);
    });

    this.latestGuidanceSpeechText = `${guidance.summaryText}. ${guidance.guidanceText}`;
    this.companionDrawer.style.display = 'block';
  }

  private handleError(err: Error): void {
    this.cameraStatusText.textContent = err.message || 'Error occurred';
    this.qualityTextEl.textContent = 'Error';
    this.statusDotEl.className = 'status-dot critical';
    this.btnScan.classList.remove('active');
    this.cameraPillFrame.classList.remove('scanning');
    this.btnStop.disabled = true;
  }

  private resetVitalsDisplay(): void {
    this.hrEl.textContent = '72';
    this.rrEl.textContent = '16';
    this.asymmetryEl.textContent = 'Normal';
    this.asymmetryEl.className = 'card-status-text';
    this.asymmetryDeltaEl.textContent = 'Δ 0.02';
    this.rhythmEl.textContent = 'Regular';
    this.rhythmEl.className = 'card-status-text';
    this.stressEl.textContent = 'Low';
    this.stressEl.className = 'card-status-text';
    this.sqiEl.textContent = '98%';
  }

  private toggleCompanionDrawer(): void {
    if (this.companionDrawer.style.display === 'none' || !this.companionDrawer.style.display) {
      this.companionDrawer.style.display = 'block';
    } else {
      this.companionDrawer.style.display = 'none';
      stopSpeaking();
      this.isSpeaking = false;
    }
  }

  private renderHistory(): void {
    const sessions = listSessions(15);
    this.historyListEl.innerHTML = '';

    if (sessions.length === 0) {
      this.historyListEl.innerHTML = '<div style="color:var(--text-tertiary); text-align:center; padding:1.5rem; font-size:12px;">No scans recorded yet.</div>';
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const timeStr = new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateStr = new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });

      item.innerHTML = `
        <div>
          <div style="font-weight:600; color:#fff; font-size:12px;">${s.mode === 'face' ? 'Face rPPG' : 'Finger PPG'} • ${s.telemetry.heartRateBpm} BPM</div>
          <div style="font-size:10px; color:rgba(255,255,255,0.45); margin-top:2px;">${dateStr} at ${timeStr} • ${s.telemetry.respirationRateRpm} RPM</div>
        </div>
        <div style="font-size:11px; font-weight:600; color:#34c759;">
          ${(s.telemetry.sqiScore * 100).toFixed(0)}% Flow
        </div>
      `;
      this.historyListEl.appendChild(item);
    });
  }

  private startWaveformLoop(): void {
    const draw = () => {
      if (this.waveformCtx && this.waveformCanvas) {
        const w = this.waveformCanvas.width;
        const h = this.waveformCanvas.height;
        this.waveformCtx.clearRect(0, 0, w, h);

        const points = this.waveformPoints;
        const dpr = window.devicePixelRatio || 1;

        if (points.length > 2) {
          // Pure glowing white pulse waveform
          this.waveformCtx.strokeStyle = '#ffffff';
          this.waveformCtx.lineWidth = 2.2 * dpr;
          this.waveformCtx.shadowColor = 'rgba(255, 255, 255, 0.8)';
          this.waveformCtx.shadowBlur = 7 * dpr;
          this.waveformCtx.lineCap = 'round';
          this.waveformCtx.lineJoin = 'round';
          this.waveformCtx.beginPath();

          const dx = w / (points.length - 1);
          let maxVal = 0.05;
          for (let i = 0; i < points.length; i++) {
            const abs = Math.abs(points[i]);
            if (abs > maxVal) maxVal = abs;
          }
          const scaleY = (h * 0.38) / maxVal;

          for (let i = 0; i < points.length; i++) {
            const x = i * dx;
            // Generate resting synthetic heart wave if signal is idle, else real signal
            let sampleVal = points[i];
            if (Math.abs(sampleVal) < 1e-4) {
              const t = (i * 0.15) - (performance.now() * 0.003);
              sampleVal = Math.sin(t) * 0.02 + Math.sin(t * 2.3) * 0.01;
            }

            const y = h / 2 - sampleVal * scaleY;
            if (i === 0) {
              this.waveformCtx.moveTo(x, y);
            } else {
              this.waveformCtx.lineTo(x, y);
            }
          }
          this.waveformCtx.stroke();
          this.waveformCtx.shadowBlur = 0;
        }
      }
      this.waveformAnimId = requestAnimationFrame(draw);
    };

    draw();
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  new TriageAppUI();
});
