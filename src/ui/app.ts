/**
 * TriageLens Web Application UI Controller
 * 
 * Binds headless TriageCoordinator, CameraStreamManager, triageStore,
 * and triageCompanion to the interactive DOM layer.
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
import { getTriageGuidance } from '../services/triageCompanion';

class TriageAppUI {
  private coordinator: TriageCoordinator;
  private currentMode: ScanMode = 'face';

  // DOM Elements
  private videoEl!: HTMLVideoElement;
  private waveformCanvas!: HTMLCanvasElement;
  private waveformCtx!: CanvasRenderingContext2D;
  private scanBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private progressBar!: HTMLElement;
  private qualityBadge!: HTMLElement;
  private faceTargetBox!: HTMLElement;
  private fingerTargetBox!: HTMLElement;

  // Metric displays
  private hrEl!: HTMLElement;
  private rrEl!: HTMLElement;
  private asymmetryEl!: HTMLElement;
  private asymmetryStatusEl!: HTMLElement;
  private sqiEl!: HTMLElement;

  // Companion displays
  private companionCard!: HTMLElement;
  private severityPill!: HTMLElement;
  private companionSummaryEl!: HTMLElement;
  private companionAdviceList!: HTMLElement;

  // Modals
  private apkModal!: HTMLElement;
  private historyModal!: HTMLElement;
  private historyListEl!: HTMLElement;

  // Waveform rendering buffer
  private waveformPoints: number[] = new Array(150).fill(0);
  private waveformAnimId: number | null = null;
  private scanStartPerfTime: number = 0;

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
    this.videoEl = document.getElementById('camera-preview') as HTMLVideoElement;
    this.waveformCanvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    this.waveformCtx = this.waveformCanvas.getContext('2d')!;
    this.scanBtn = document.getElementById('btn-start-scan') as HTMLButtonElement;
    this.stopBtn = document.getElementById('btn-stop-scan') as HTMLButtonElement;
    this.progressBar = document.getElementById('scan-progress-bar') as HTMLElement;
    this.qualityBadge = document.getElementById('quality-badge') as HTMLElement;
    this.faceTargetBox = document.getElementById('face-target-box') as HTMLElement;
    this.fingerTargetBox = document.getElementById('finger-target-box') as HTMLElement;

    this.hrEl = document.getElementById('val-heart-rate') as HTMLElement;
    this.rrEl = document.getElementById('val-respiration') as HTMLElement;
    this.asymmetryEl = document.getElementById('val-asymmetry') as HTMLElement;
    this.asymmetryStatusEl = document.getElementById('status-asymmetry') as HTMLElement;
    this.sqiEl = document.getElementById('val-sqi') as HTMLElement;

    this.companionCard = document.getElementById('companion-card') as HTMLElement;
    this.severityPill = document.getElementById('severity-pill') as HTMLElement;
    this.companionSummaryEl = document.getElementById('companion-summary') as HTMLElement;
    this.companionAdviceList = document.getElementById('companion-advice-list') as HTMLElement;

    this.apkModal = document.getElementById('apk-modal') as HTMLElement;
    this.historyModal = document.getElementById('history-modal') as HTMLElement;
    this.historyListEl = document.getElementById('history-list') as HTMLElement;

    // Mode Toggle buttons
    const btnModeFace = document.getElementById('mode-btn-face') as HTMLButtonElement;
    const btnModeFinger = document.getElementById('mode-btn-finger') as HTMLButtonElement;

    btnModeFace.addEventListener('click', () => {
      this.switchMode('face');
      btnModeFace.classList.add('active');
      btnModeFinger.classList.remove('active');
    });

    btnModeFinger.addEventListener('click', () => {
      this.switchMode('finger');
      btnModeFinger.classList.add('active');
      btnModeFace.classList.remove('active');
    });

    // Control buttons
    this.scanBtn.addEventListener('click', () => this.startScan());
    this.stopBtn.addEventListener('click', () => this.stopScan());

    // Modal triggers
    document.getElementById('nav-btn-apk')?.addEventListener('click', () => {
      this.apkModal.classList.add('open');
    });
    document.getElementById('close-apk-modal')?.addEventListener('click', () => {
      this.apkModal.classList.remove('open');
    });

    document.getElementById('nav-btn-history')?.addEventListener('click', () => {
      this.renderHistory();
      this.historyModal.classList.add('open');
    });
    document.getElementById('close-history-modal')?.addEventListener('click', () => {
      this.historyModal.classList.remove('open');
    });

    document.getElementById('btn-clear-history')?.addEventListener('click', () => {
      clearSessions();
      this.renderHistory();
    });

    // Resize canvas
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    window.addEventListener('beforeunload', () => {
      if (this.waveformAnimId !== null) {
        cancelAnimationFrame(this.waveformAnimId);
      }
    });
  }

  private resizeCanvas(): void {
    if (this.waveformCanvas) {
      this.waveformCanvas.width = this.waveformCanvas.clientWidth * window.devicePixelRatio || 400;
      this.waveformCanvas.height = this.waveformCanvas.clientHeight * window.devicePixelRatio || 100;
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
      this.qualityBadge.textContent = 'Face Mode Ready';
    } else {
      this.videoEl.classList.add('finger-mode');
      this.faceTargetBox.style.display = 'none';
      this.fingerTargetBox.style.display = 'flex';
      this.qualityBadge.textContent = 'Finger Mode Ready';
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
    this.scanBtn.disabled = true;
    this.stopBtn.disabled = false;
    this.resetVitalsDisplay();
    this.progressBar.style.width = '0%';
    this.scanStartPerfTime = performance.now();

    try {
      await this.coordinator.startScan(this.currentMode, this.videoEl);
    } catch (err: any) {
      this.qualityBadge.textContent = `Error: ${err.message}`;
      this.qualityBadge.className = 'quality-badge danger';
      this.scanBtn.disabled = false;
      this.stopBtn.disabled = true;
    }
  }

  private stopScan(): void {
    this.coordinator.stopProcessing();
    this.scanBtn.disabled = false;
    this.stopBtn.disabled = true;
    this.qualityBadge.textContent = 'Scan Stopped';
    this.qualityBadge.className = 'quality-badge';
    this.progressBar.style.width = '0%';
  }

  private handleStateChange(state: TriageState): void {
    switch (state) {
      case 'STARTING':
        this.qualityBadge.textContent = 'Initializing Camera...';
        this.qualityBadge.className = 'quality-badge';
        break;
      case 'CALIBRATING':
        this.qualityBadge.textContent = 'Stabilizing Sensor (2s)...';
        this.qualityBadge.className = 'quality-badge warning';
        break;
      case 'ACQUIRING':
        this.qualityBadge.textContent = 'Acquiring Vitals (15s)...';
        this.qualityBadge.className = 'quality-badge';
        this.scanStartPerfTime = performance.now();
        break;
      case 'ANALYZING':
        this.qualityBadge.textContent = 'Processing Spectral DSP...';
        this.qualityBadge.className = 'quality-badge';
        this.progressBar.style.width = '100%';
        break;
      case 'COMPLETE':
        this.qualityBadge.textContent = 'Telemetry Complete';
        this.qualityBadge.className = 'quality-badge';
        this.scanBtn.disabled = false;
        this.stopBtn.disabled = true;
        break;
      case 'FAILED':
        this.qualityBadge.textContent = 'Scan Interrupted / Failed';
        this.qualityBadge.className = 'quality-badge danger';
        this.scanBtn.disabled = false;
        this.stopBtn.disabled = true;
        break;
    }
  }

  private handleWaveformSample(sample: number): void {
    this.waveformPoints.push(sample);
    if (this.waveformPoints.length > 200) {
      this.waveformPoints.shift();
    }

    // Update real-time progress bar if acquiring
    if (this.coordinator.getState() === 'ACQUIRING') {
      const elapsed = performance.now() - this.scanStartPerfTime;
      const pct = Math.min(100, Math.max(0, (elapsed / 15000) * 100));
      this.progressBar.style.width = `${pct}%`;
    }
  }

  private handleQualityDiagnostic(diag: QualityDiagnostic): void {
    if (this.currentMode === 'face') {
      this.faceTargetBox.classList.toggle('active', diag.isAcceptable);
    } else {
      this.fingerTargetBox.classList.toggle('active', diag.isAcceptable);
    }

    if (!diag.isAcceptable) {
      if (diag.reason === 'FINGER_OFF') {
        this.qualityBadge.textContent = 'Cover Camera Completely';
      } else if (diag.reason === 'MOTION_ARTIFACT') {
        this.qualityBadge.textContent = 'Keep Face Centered & Still';
      } else if (diag.reason === 'LOW_LIGHT') {
        this.qualityBadge.textContent = 'Low Light - Move Closer';
      } else if (diag.reason === 'POOR_SNR') {
        this.qualityBadge.textContent = 'High Optical Noise';
      }
      this.qualityBadge.className = 'quality-badge warning';
    }
  }

  private async handleScanComplete(telemetry: VitalsTelemetry): Promise<void> {
    // 1. Display verified telemetry numbers
    this.hrEl.textContent = telemetry.heartRateBpm > 0 ? telemetry.heartRateBpm.toFixed(0) : '--';
    this.rrEl.textContent = telemetry.respirationRateRpm > 0 ? telemetry.respirationRateRpm.toFixed(0) : '--';
    this.sqiEl.textContent = `${(telemetry.sqiScore * 100).toFixed(0)}%`;

    if (this.currentMode === 'face') {
      this.asymmetryEl.textContent = telemetry.strokeAsymmetryDelta.toFixed(3);
      if (telemetry.strokeAsymmetryDelta >= 0.12) {
        this.asymmetryStatusEl.textContent = 'ALERT: FAST Asymmetry';
        this.asymmetryStatusEl.style.color = '#ef4444';
      } else if (telemetry.strokeAsymmetryDelta >= 0.06) {
        this.asymmetryStatusEl.textContent = 'Mild Asymmetry';
        this.asymmetryStatusEl.style.color = '#f59e0b';
      } else {
        this.asymmetryStatusEl.textContent = 'Normal Symmetry';
        this.asymmetryStatusEl.style.color = '#10b981';
      }
    } else {
      this.asymmetryEl.textContent = 'N/A';
      this.asymmetryStatusEl.textContent = 'Finger Mode';
      this.asymmetryStatusEl.style.color = 'var(--text-muted)';
    }

    // 2. Compare against baseline and persist to local storage
    const previousSessions = listSessions(1);
    const baseline = previousSessions.length > 0 ? previousSessions[0] : null;
    const delta = compareWithBaseline(telemetry, baseline);

    saveSession({
      mode: this.currentMode,
      telemetry,
      durationMs: 15000,
    });

    // 3. Generate AI Companion clinical guidance
    this.companionCard.style.display = 'flex';
    this.companionSummaryEl.textContent = 'Synthesizing clinical interpretation...';

    const guidance = await getTriageGuidance({
      telemetry,
      mode: this.currentMode,
      baselineDelta: delta,
    });

    this.severityPill.textContent = guidance.severityLevel;
    this.severityPill.className = `severity-pill ${guidance.severityLevel}`;
    this.companionSummaryEl.textContent = guidance.summaryText + ' ' + guidance.guidanceText;

    this.companionAdviceList.innerHTML = '';
    guidance.actionableAdvice.forEach((advice) => {
      const li = document.createElement('li');
      li.textContent = advice;
      this.companionAdviceList.appendChild(li);
    });
  }

  private handleError(err: Error): void {
    this.qualityBadge.textContent = err.message || 'Error occurred';
    this.qualityBadge.className = 'quality-badge danger';
    this.scanBtn.disabled = false;
    this.stopBtn.disabled = true;
  }

  private resetVitalsDisplay(): void {
    this.hrEl.textContent = '--';
    this.rrEl.textContent = '--';
    this.asymmetryEl.textContent = '--';
    this.asymmetryStatusEl.textContent = 'Ready';
    this.asymmetryStatusEl.style.color = 'var(--text-muted)';
    this.sqiEl.textContent = '--%';
    this.companionCard.style.display = 'none';
  }

  private renderHistory(): void {
    const sessions = listSessions(15);
    this.historyListEl.innerHTML = '';

    if (sessions.length === 0) {
      this.historyListEl.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:1.5rem;">No scans recorded yet.</div>';
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const timeStr = new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const dateStr = new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });

      item.innerHTML = `
        <div>
          <div style="font-weight:600; color:#fff;">${s.mode.toUpperCase()} SCAN • ${s.telemetry.heartRateBpm} BPM</div>
          <div style="font-size:0.7rem; color:var(--text-muted);">${dateStr} at ${timeStr} • RR: ${s.telemetry.respirationRateRpm} RPM</div>
        </div>
        <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--accent-green);">
          SQI ${(s.telemetry.sqiScore * 100).toFixed(0)}%
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

        // Center line
        this.waveformCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.waveformCtx.lineWidth = 1;
        this.waveformCtx.beginPath();
        this.waveformCtx.moveTo(0, h / 2);
        this.waveformCtx.lineTo(w, h / 2);
        this.waveformCtx.stroke();

        // Waveform trace
        const points = this.waveformPoints;
        if (points.length > 2) {
          this.waveformCtx.strokeStyle = '#10b981';
          this.waveformCtx.lineWidth = 2.5 * window.devicePixelRatio;
          this.waveformCtx.shadowColor = '#10b981';
          this.waveformCtx.shadowBlur = 8;
          this.waveformCtx.beginPath();

          const dx = w / (points.length - 1);
          // Auto-scale peak-to-peak amplitude
          let maxVal = 0.05;
          for (let i = 0; i < points.length; i++) {
            const abs = Math.abs(points[i]);
            if (abs > maxVal) maxVal = abs;
          }
          const scaleY = (h * 0.4) / maxVal;

          for (let i = 0; i < points.length; i++) {
            const x = i * dx;
            const y = h / 2 - points[i] * scaleY;
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

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  new TriageAppUI();
});
