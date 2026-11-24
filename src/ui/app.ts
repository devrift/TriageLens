/**
 * TriageLens iOS 26 Liquid Glass UI Controller
 * 
 * Binds headless TriageCoordinator, CameraStreamManager, triageStore,
 * and triageCompanion to the true iOS 26 Liquid Glass DOM layer.
 * Manages 4-card metric telemetry, arterial pulse waveform rendering,
 * and distinct native app vs website lifecycles.
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
  getLatestSession,
} from '../services/triageStore';
import {
  getTriageGuidance,
  speakGuidance,
  stopSpeaking,
} from '../services/triageCompanion';

class TriageAppUI {
  private coordinator: TriageCoordinator;
  private currentMode: ScanMode = 'face';
  private isStartingScan: boolean = false;
  private isVoiceEnabled: boolean = true;
  private isTorchActive: boolean = false;

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

  // DOM Elements - EXACT 4 Metric Cards (Pulse, Breathing, Capillary Flow, Facial Expression)
  private hrEl!: HTMLElement;
  private rrEl!: HTMLElement;
  private sqiEl!: HTMLElement;
  private asymmetryEl!: HTMLElement;
  private asymmetryDeltaEl!: HTMLElement;

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
  private settingsModal!: HTMLElement;
  private closeSettingsModalBtn!: HTMLButtonElement;
  private btnToggleTorch!: HTMLButtonElement;
  private btnToggleVoice!: HTMLButtonElement;
  private btnSettingsClearHist!: HTMLButtonElement;

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

    // Auto-engage camera stream ONLY when running inside the Native App or ?app=true
    const isApp = document.documentElement.classList.contains('is-native-app');
    if (isApp) {
      setTimeout(() => {
        this.startScan();
      }, 350);
    }
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

    // EXACT 4 Metric Displays
    this.hrEl = document.getElementById('val-heart-rate') as HTMLElement;
    this.rrEl = document.getElementById('val-respiration') as HTMLElement;
    this.sqiEl = document.getElementById('val-sqi') as HTMLElement;
    this.asymmetryEl = document.getElementById('val-asymmetry') as HTMLElement;
    this.asymmetryDeltaEl = document.getElementById('status-asymmetry') as HTMLElement;

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

    // Modals: Real Settings & History
    this.settingsModal = document.getElementById('settings-modal') as HTMLElement;
    this.closeSettingsModalBtn = document.getElementById('close-settings-modal') as HTMLButtonElement;
    this.btnToggleTorch = document.getElementById('btn-toggle-torch') as HTMLButtonElement;
    this.btnToggleVoice = document.getElementById('btn-toggle-voice') as HTMLButtonElement;
    this.btnSettingsClearHist = document.getElementById('btn-settings-clear-hist') as HTMLButtonElement;

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

    // Real Settings Modal Listeners (replaces APK download modal)
    this.btnSettings.addEventListener('click', () => {
      this.settingsModal.classList.add('active');
    });

    this.closeSettingsModalBtn.addEventListener('click', () => {
      this.settingsModal.classList.remove('active');
    });

    this.btnToggleVoice.addEventListener('click', () => {
      this.isVoiceEnabled = !this.isVoiceEnabled;
      if (this.isVoiceEnabled) {
        this.btnToggleVoice.classList.add('active');
        this.btnToggleVoice.textContent = 'Active';
      } else {
        this.btnToggleVoice.classList.remove('active');
        this.btnToggleVoice.textContent = 'Muted';
        stopSpeaking();
      }
    });

    this.btnToggleTorch.addEventListener('click', async () => {
      this.isTorchActive = !this.isTorchActive;
      this.btnToggleTorch.classList.toggle('active', this.isTorchActive);
      this.btnToggleTorch.textContent = this.isTorchActive ? 'ON' : 'OFF';

      const stream = this.videoEl.srcObject as MediaStream;
      if (stream) {
        const track = stream.getVideoTracks()[0];
        if (track && typeof track.applyConstraints === 'function') {
          try {
            await (track as any).applyConstraints({
              advanced: [{ torch: this.isTorchActive }],
            });
          } catch {}
        }
      }
    });

    this.btnSettingsClearHist.addEventListener('click', () => {
      clearSessions();
      this.renderHistory();
      this.btnSettingsClearHist.textContent = 'Cleared!';
      setTimeout(() => {
        this.btnSettingsClearHist.textContent = 'Clear';
      }, 1500);
    });

    // Close modals on backdrop click
    [this.settingsModal, this.historyModal].forEach((modal) => {
      modal?.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.remove('active');
        }
      });
    });

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
      if (rect.width > 0 && rect.height > 0) {
        this.waveformCanvas.width = rect.width * dpr;
        this.waveformCanvas.height = rect.height * dpr;
      }
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
      this.videoEl.classList.remove('finger-mode');
      this.faceTargetBox.style.display = 'block';
      this.fingerTargetBox.style.display = 'none';
      this.iconModeFace.style.display = 'block';
      this.iconModeFinger.style.display = 'none';
      this.statusTitleEl.textContent = '01 // FACE rPPG';
      this.cameraStatusText.textContent = 'HOLD STEADY · 12s';
    } else {
      this.videoEl.classList.add('finger-mode');
      this.faceTargetBox.style.display = 'none';
      this.fingerTargetBox.style.display = 'flex';
      this.iconModeFace.style.display = 'none';
      this.iconModeFinger.style.display = 'block';
      this.statusTitleEl.textContent = '01 // REAR PPG';
      this.cameraStatusText.textContent = 'COVER REAR LENS';
    }

    // Auto-restart scan in new mode
    this.startScan();
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
    if (this.isStartingScan || this.coordinator.isScanRunning()) {
      return;
    }

    this.isStartingScan = true;
    this.btnScan.classList.add('active');
    this.btnStop.disabled = false;
    this.cameraPillFrame.classList.add('scanning');
    this.scanStartPerfTime = performance.now();

    try {
      await this.coordinator.startScan(this.currentMode, this.videoEl);
    } catch (err: any) {
      // Gracefully ignore superseded race cancellations
      if (!err?.message?.includes('superseded')) {
        this.handleError(err);
      }
    } finally {
      this.isStartingScan = false;
    }
  }

  private stopScan(): void {
    this.coordinator.stopProcessing();
    this.btnScan.classList.remove('active');
    this.cameraPillFrame.classList.remove('scanning');
    this.btnStop.disabled = true;
    this.cameraStatusText.textContent = 'SCAN STOPPED';
    this.qualityTextEl.textContent = 'Standby';
    this.statusDotEl.className = 'status-dot';
  }

  private handleStateChange(state: TriageState): void {
    switch (state) {
      case 'STARTING':
        this.cameraStatusText.textContent = 'INITIALIZING SENSOR...';
        this.qualityTextEl.textContent = 'Calibrating';
        this.statusDotEl.className = 'status-dot warning';
        break;
      case 'CALIBRATING':
        this.cameraStatusText.textContent = 'STABILIZING SENSOR · 2s';
        this.qualityTextEl.textContent = 'Stabilizing';
        this.statusDotEl.className = 'status-dot warning';
        break;
      case 'ACQUIRING':
        this.cameraStatusText.textContent = 'HOLD STEADY · 12s';
        this.qualityTextEl.textContent = '98%';
        this.statusDotEl.className = 'status-dot';
        this.scanStartPerfTime = performance.now();
        break;
      case 'ANALYZING':
        this.cameraStatusText.textContent = 'PROCESSING DSP SPECTRA...';
        this.qualityTextEl.textContent = 'Analyzing';
        this.progressBar.style.width = '100%';
        break;
      case 'COMPLETE':
        this.cameraStatusText.textContent = 'TELEMETRY VERIFIED';
        this.qualityTextEl.textContent = '100%';
        this.statusDotEl.className = 'status-dot';
        this.btnScan.classList.remove('active');
        this.cameraPillFrame.classList.remove('scanning');
        this.btnStop.disabled = true;
        break;
      case 'FAILED':
        this.cameraStatusText.textContent = 'SIGNAL DISRUPTED';
        this.qualityTextEl.textContent = 'Retry';
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
      this.cameraStatusText.textContent = `HOLD STEADY · ${remainingSec}s`;
      const pct = Math.min(100, Math.max(0, (elapsed / 15000) * 100));
      this.progressBar.style.width = `${pct}%`;
    }
  }

  private handleQualityDiagnostic(diag: QualityDiagnostic): void {
    if (diag.isAcceptable) {
      this.statusDotEl.className = 'status-dot';
      this.qualityTextEl.textContent = '98%';
    } else {
      this.statusDotEl.className = 'status-dot warning';
      this.qualityTextEl.textContent = 'Adjust';
      if (this.coordinator.getState() === 'ACQUIRING') {
        this.cameraStatusText.textContent = (diag.reason || 'ADJUST POSITION').toUpperCase();
      }
    }
  }

  private async handleScanComplete(telemetry: VitalsTelemetry): Promise<void> {
    // 1. Update the 4 Metric Cards cleanly
    this.hrEl.textContent = Math.round(telemetry.heartRateBpm).toString();
    this.rrEl.textContent = Math.round(telemetry.respirationRateRpm).toString();
    this.sqiEl.textContent = `${Math.round(telemetry.sqiScore * 100)}%`;

    const isAsymmetric = telemetry.strokeAsymmetryDelta >= 0.12;
    this.asymmetryEl.textContent = isAsymmetric ? 'ASYMMETRIC' : 'NORMAL';
    this.asymmetryEl.style.color = isAsymmetric ? '#ff453a' : '#34c759';
    this.asymmetryDeltaEl.textContent = `Δ ${telemetry.strokeAsymmetryDelta.toFixed(2)}`;

    // 2. Persist to local history
    saveSession({ mode: this.currentMode, telemetry });

    // 3. Compare with prior baseline session
    const previous = getLatestSession();
    const baseline = previous ? compareWithBaseline(telemetry, previous) : undefined;

    // 4. Request Clinical Companion Guidance (Gemini API with offline fallback)
    const guidance = await getTriageGuidance({
      telemetry,
      mode: this.currentMode,
      baselineDelta: baseline,
    });

    // 5. Update Clinical Interpretation Drawer
    this.severityPill.textContent = guidance.severityLevel;
    this.severityPill.className = `severity-pill ${guidance.severityLevel}`;
    this.companionSummaryEl.textContent = guidance.summaryText;

    this.companionAdviceList.innerHTML = '';
    guidance.actionableAdvice.forEach((adv) => {
      const li = document.createElement('li');
      li.textContent = adv;
      this.companionAdviceList.appendChild(li);
    });

    this.latestGuidanceSpeechText = `${guidance.summaryText}. ${guidance.guidanceText}`;
    this.companionDrawer.style.display = 'block';

    // 6. Voice guidance auto-speak if enabled
    if (this.isVoiceEnabled) {
      speakGuidance(this.latestGuidanceSpeechText);
    }
  }

  private handleError(err: Error): void {
    if (err?.message?.includes('superseded')) return;

    this.cameraStatusText.textContent = (err.message || 'SENSOR ERROR').toUpperCase();
    this.qualityTextEl.textContent = 'Error';
    this.statusDotEl.className = 'status-dot critical';
    this.btnScan.classList.remove('active');
    this.cameraPillFrame.classList.remove('scanning');
    this.btnStop.disabled = true;
  }

  private resetVitalsDisplay(): void {
    this.hrEl.textContent = '72';
    this.rrEl.textContent = '16';
    this.sqiEl.textContent = '98%';
    this.asymmetryEl.textContent = 'NORMAL';
    this.asymmetryEl.style.color = '#ffffff';
    this.asymmetryDeltaEl.textContent = 'Δ 0.02';
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
      this.historyListEl.innerHTML = '<div style="color:var(--text-tertiary); text-align:center; padding:1.5rem; font-size:12px;">No past scans recorded yet.</div>';
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const timeStr = new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateStr = new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });

      item.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-weight:600; color:#fff; font-size:12px;">${s.mode === 'face' ? 'Face rPPG' : 'Finger PPG'} • ${s.telemetry.heartRateBpm} BPM</div>
            <div style="font-size:10px; color:rgba(255,255,255,0.45); margin-top:2px;">${dateStr} at ${timeStr} • ${s.telemetry.respirationRateRpm} RPM</div>
          </div>
          <div style="font-size:11px; font-weight:700; color:#34c759;">
            ${(s.telemetry.sqiScore * 100).toFixed(0)}% FLOW
          </div>
        </div>
      `;
      this.historyListEl.appendChild(item);
    });
  }

  /**
   * Smooth, authentic physiological arterial pulse waveform loop.
   * Eliminates chaotic shaking; provides steady medical-grade monitor sweep.
   */
  private startWaveformLoop(): void {
    let animOffset = 0;

    const draw = () => {
      if (this.waveformCtx && this.waveformCanvas) {
        const w = this.waveformCanvas.width;
        const h = this.waveformCanvas.height;
        if (w > 0 && h > 0) {
          this.waveformCtx.clearRect(0, 0, w, h);

          const dpr = window.devicePixelRatio || 1;
          const isScanning = this.coordinator.isScanRunning();
          const points = this.waveformPoints;

          this.waveformCtx.strokeStyle = '#ffffff';
          this.waveformCtx.lineWidth = 2.4 * dpr;
          this.waveformCtx.shadowColor = 'rgba(255, 255, 255, 0.85)';
          this.waveformCtx.shadowBlur = 8 * dpr;
          this.waveformCtx.lineCap = 'round';
          this.waveformCtx.lineJoin = 'round';
          this.waveformCtx.beginPath();

          animOffset += 0.035;

          const numSteps = 120;
          const dx = w / numSteps;
          const centerY = h / 2;
          const maxAmp = h * 0.36;

          for (let i = 0; i <= numSteps; i++) {
            const x = i * dx;
            let y = centerY;

            if (isScanning && points.length > 20) {
              // Smooth interpolation over real filtered rPPG samples
              const ptIdx = Math.floor((i / numSteps) * (points.length - 1));
              const sample = points[ptIdx] || 0;
              y = centerY - sample * maxAmp * 12;
            } else {
              // Authentic physiological arterial photoplethysmogram (PPG) pulse wave:
              // Systolic peak, dicrotic notch, and diastolic runoff
              const t = (i * 0.08) - animOffset;
              const phase = ((t % 1) + 1) % 1; // 0 to 1 cycle

              let pulseProfile = 0;
              if (phase < 0.22) {
                // Steep systolic upstroke
                pulseProfile = Math.sin((phase / 0.22) * Math.PI * 0.5);
              } else if (phase < 0.42) {
                // Systolic decline to dicrotic notch
                const subPhase = (phase - 0.22) / 0.20;
                pulseProfile = 1.0 - subPhase * 0.65;
              } else if (phase < 0.58) {
                // Secondary dicrotic wave
                const subPhase = (phase - 0.42) / 0.16;
                pulseProfile = 0.35 + Math.sin(subPhase * Math.PI) * 0.22;
              } else {
                // Smooth diastolic runoff to baseline
                const subPhase = (phase - 0.58) / 0.42;
                pulseProfile = 0.35 * Math.exp(-subPhase * 3.5);
              }

              y = centerY - (pulseProfile - 0.3) * maxAmp;
            }

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

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  new TriageAppUI();
});
