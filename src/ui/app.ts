/**
 * TriageLens Native & `/app` Mobile Scanner Controller
 * 
 * Strict Zero-Mock Implementation:
 *   - All telemetry cards display `N/A` until real optical signal acquisition converges.
 *   - Waveform renders a flat zero baseline with a subtle idle sweep line when unscanned.
 *   - Enforces route & environment isolation (`Capacitor.isNativePlatform()` or `/app`).
 *   - Coordinates 4 functional navigation tabs: Scanner, Mode, History, Settings & Diagnostics.
 */

import { Capacitor } from '@capacitor/core';
import {
  LiveTelemetrySnapshot,
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

type AppTab = 'scanner' | 'mode' | 'history' | 'settings';

class TriageAppUI {
  private coordinator: TriageCoordinator;
  private currentMode: ScanMode = 'face';
  private isStartingScan: boolean = false;
  private isVoiceEnabled: boolean = true;
  private isTorchPreferred: boolean = false;

  // Live telemetry freeze & 1.5s quality-drop reset state
  private isCalculationFrozen: boolean = false;
  private qualityDropTimer: ReturnType<typeof setTimeout> | null = null;
  private hasActiveSignalSamples: boolean = false;

  // FPS measurement
  private frameSampleCount: number = 0;
  private lastFpsUpdatePerf: number = 0;
  private measuredFps: number = 0;

  // Top Bar DOM
  private qualityPillEl!: HTMLElement;
  private qualityTextEl!: HTMLElement;
  private statusDotEl!: HTMLElement;

  // Tab Views & Navbar Buttons
  private tabViews!: Record<AppTab, HTMLElement>;
  private navTabBtns!: Record<AppTab, HTMLButtonElement>;

  // Scanner Tab DOM
  private cameraPillFrame!: HTMLElement;
  private videoEl!: HTMLVideoElement;
  private faceTargetBox!: HTMLElement;
  private fingerTargetBox!: HTMLElement;
  private cameraStatusText!: HTMLElement;
  private progressBar!: HTMLElement;
  private waveformModeLabel!: HTMLElement;
  private waveformCanvas!: HTMLCanvasElement;
  private waveformCtx!: CanvasRenderingContext2D;

  // 6 Metric Cards DOM
  private hrEl!: HTMLElement;
  private hrUnitEl!: HTMLElement;
  private rrEl!: HTMLElement;
  private rrUnitEl!: HTMLElement;
  private sqiEl!: HTMLElement;
  private sqiUnitEl!: HTMLElement;
  private asymmetryEl!: HTMLElement;
  private asymmetrySubEl!: HTMLElement;
  private rhythmEl!: HTMLElement;
  private rhythmSubEl!: HTMLElement;
  private stressEl!: HTMLElement;
  private stressSubEl!: HTMLElement;
  private pulseIconEl!: SVGElement | null;

  // Scanner Actions
  private btnPrimaryScan!: HTMLButtonElement;
  private btnOpenInsights!: HTMLButtonElement;

  // Mode Tab DOM
  private modeCardFace!: HTMLButtonElement;
  private modeCardFinger!: HTMLButtonElement;
  private badgeModeFace!: HTMLElement;
  private badgeModeFinger!: HTMLElement;
  private btnModeTorch!: HTMLButtonElement;
  private modeTorchStatus!: HTMLElement;
  private btnModeGoScan!: HTMLButtonElement;

  // History Tab DOM
  private historyListEl!: HTMLElement;
  private btnExportHistory!: HTMLButtonElement;
  private btnClearHistory!: HTMLButtonElement;

  // Settings & Diagnostics Tab DOM
  private resBtns!: NodeListOf<HTMLButtonElement>;
  private diagFpsEl!: HTMLElement;
  private diagCameraResEl!: HTMLElement;
  private diagTorchSupportEl!: HTMLElement;
  private diagOfflineModelsEl!: HTMLElement;
  private diagPermissionEl!: HTMLElement;
  private btnToggleTorch!: HTMLButtonElement;
  private btnToggleVoice!: HTMLButtonElement;
  private btnRequestPermission!: HTMLButtonElement;

  // Clinical Interpretation Drawer DOM
  private companionDrawer!: HTMLElement;
  private severityPill!: HTMLElement;
  private companionSummaryEl!: HTMLElement;
  private companionAdviceList!: HTMLElement;
  private btnSpeakGuidance!: HTMLButtonElement;
  private closeDrawerBtn!: HTMLButtonElement;
  private latestGuidanceSpeechText: string = '';

  // Waveform rolling buffer (only populated with real filtered samples)
  private readonly waveformPoints: number[] = [];
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
      minLumaThreshold: 80,
    });

    const isAppContext = this.isAppShellActive();
    if (!isAppContext) {
      // Landing page only: do not start camera, do not attach canvas loop
      return;
    }

    // Strip landing markup completely inside Native APK or `/app`
    document.documentElement.classList.add('is-native-app');
    document.getElementById('landing-view')?.remove();

    this.initDom();
    this.initCoordinatorCallbacks();
    this.setIdleStateDisplay();
    this.refreshHistory();
    this.refreshDiagnostics();
    this.startWaveformLoop();
  }

  private isAppShellActive(): boolean {
    let capNative = false;
    try {
      capNative = typeof Capacitor?.isNativePlatform === 'function' && Capacitor.isNativePlatform();
    } catch {}

    const isProtoNative =
      window.location.protocol === 'capacitor:' || window.location.protocol === 'ionic:';
    const isAndroidWebView =
      /Android/i.test(navigator.userAgent) &&
      (/; wv\)/i.test(navigator.userAgent) || window.location.hostname === 'localhost');
    const isAppRoute =
      window.location.pathname === '/app' ||
      window.location.pathname.startsWith('/app/') ||
      window.location.search.includes('app=true') ||
      window.location.hash === '#app';

    return (
      capNative ||
      isProtoNative ||
      isAndroidWebView ||
      isAppRoute ||
      document.documentElement.classList.contains('is-native-app')
    );
  }

  private initDom(): void {
    // Top bar
    this.qualityPillEl = document.getElementById('quality-pill') as HTMLElement;
    this.qualityTextEl = document.getElementById('quality-text') as HTMLElement;
    this.statusDotEl = this.qualityPillEl.querySelector('.status-dot') as HTMLElement;

    // Tab views & nav buttons
    this.tabViews = {
      scanner: document.getElementById('view-scanner') as HTMLElement,
      mode: document.getElementById('view-mode') as HTMLElement,
      history: document.getElementById('view-history') as HTMLElement,
      settings: document.getElementById('view-settings') as HTMLElement,
    };

    this.navTabBtns = {
      scanner: document.getElementById('nav-tab-scanner') as HTMLButtonElement,
      mode: document.getElementById('nav-tab-mode') as HTMLButtonElement,
      history: document.getElementById('nav-tab-history') as HTMLButtonElement,
      settings: document.getElementById('nav-tab-settings') as HTMLButtonElement,
    };

    (Object.keys(this.navTabBtns) as AppTab[]).forEach((tab) => {
      this.navTabBtns[tab].addEventListener('click', () => this.switchTab(tab));
    });

    // Scanner tab elements
    this.cameraPillFrame = document.querySelector('.camera-pill-frame') as HTMLElement;
    this.videoEl = document.getElementById('camera-preview') as HTMLVideoElement;
    this.faceTargetBox = document.getElementById('face-target-box') as HTMLElement;
    this.fingerTargetBox = document.getElementById('finger-target-box') as HTMLElement;
    this.cameraStatusText = document.getElementById('camera-status-text') as HTMLElement;
    this.progressBar = document.getElementById('scan-progress-bar') as HTMLElement;
    this.waveformModeLabel = document.getElementById('waveform-mode-label') as HTMLElement;
    this.waveformCanvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    this.waveformCtx = this.waveformCanvas.getContext('2d')!;

    // 6 Metric cards
    this.hrEl = document.getElementById('val-heart-rate') as HTMLElement;
    this.hrUnitEl = document.getElementById('unit-heart-rate') as HTMLElement;
    this.rrEl = document.getElementById('val-respiration') as HTMLElement;
    this.rrUnitEl = document.getElementById('unit-respiration') as HTMLElement;
    this.sqiEl = document.getElementById('val-sqi') as HTMLElement;
    this.sqiUnitEl = document.getElementById('unit-sqi') as HTMLElement;
    this.asymmetryEl = document.getElementById('val-asymmetry') as HTMLElement;
    this.asymmetrySubEl = document.getElementById('status-asymmetry') as HTMLElement;
    this.rhythmEl = document.getElementById('val-rhythm') as HTMLElement;
    this.rhythmSubEl = document.getElementById('status-rhythm') as HTMLElement;
    this.stressEl = document.getElementById('val-stress') as HTMLElement;
    this.stressSubEl = document.getElementById('status-stress') as HTMLElement;
    this.pulseIconEl = document.querySelector('#card-pulse .card-icon');

    this.btnPrimaryScan = document.getElementById('btn-primary-scan') as HTMLButtonElement;
    this.btnOpenInsights = document.getElementById('btn-open-insights') as HTMLButtonElement;

    this.btnPrimaryScan.addEventListener('click', () => {
      if (this.coordinator.isScanRunning() || this.isStartingScan) {
        this.stopScan();
      } else {
        this.startScan();
      }
    });

    this.btnOpenInsights.addEventListener('click', () => {
      this.companionDrawer.style.display = 'block';
    });

    // Mode tab elements
    this.modeCardFace = document.getElementById('mode-card-face') as HTMLButtonElement;
    this.modeCardFinger = document.getElementById('mode-card-finger') as HTMLButtonElement;
    this.badgeModeFace = document.getElementById('badge-mode-face') as HTMLElement;
    this.badgeModeFinger = document.getElementById('badge-mode-finger') as HTMLElement;
    this.btnModeTorch = document.getElementById('btn-mode-torch') as HTMLButtonElement;
    this.modeTorchStatus = document.getElementById('mode-torch-status') as HTMLElement;
    this.btnModeGoScan = document.getElementById('btn-mode-go-scan') as HTMLButtonElement;

    this.modeCardFace.addEventListener('click', () => this.selectMode('face', false));
    this.modeCardFinger.addEventListener('click', () => this.selectMode('finger', false));
    this.btnModeGoScan.addEventListener('click', () => {
      this.switchTab('scanner');
      this.startScan();
    });

    this.btnModeTorch.addEventListener('click', () => this.toggleTorch());

    // History tab elements
    this.historyListEl = document.getElementById('history-list') as HTMLElement;
    this.btnExportHistory = document.getElementById('btn-export-history') as HTMLButtonElement;
    this.btnClearHistory = document.getElementById('btn-clear-history') as HTMLButtonElement;

    this.btnClearHistory.addEventListener('click', () => {
      clearSessions();
      this.refreshHistory();
    });

    this.btnExportHistory.addEventListener('click', () => this.exportHistoryJson());

    // Settings & Diagnostics tab elements
    this.resBtns = document.querySelectorAll('#resolution-group .res-btn');
    this.diagFpsEl = document.getElementById('diag-fps') as HTMLElement;
    this.diagCameraResEl = document.getElementById('diag-camera-res') as HTMLElement;
    this.diagTorchSupportEl = document.getElementById('diag-torch-support') as HTMLElement;
    this.diagOfflineModelsEl = document.getElementById('diag-offline-models') as HTMLElement;
    this.diagPermissionEl = document.getElementById('diag-permission') as HTMLElement;
    this.btnToggleTorch = document.getElementById('btn-toggle-torch') as HTMLButtonElement;
    this.btnToggleVoice = document.getElementById('btn-toggle-voice') as HTMLButtonElement;
    this.btnRequestPermission = document.getElementById('btn-request-permission') as HTMLButtonElement;

    this.resBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const resStr = btn.getAttribute('data-res') || '640x480';
        const [wStr, hStr] = resStr.split('x');
        const width = parseInt(wStr, 10) || 640;
        const height = parseInt(hStr, 10) || 480;

        this.resBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.coordinator.setCameraResolution(width, height);
        this.refreshDiagnostics();

        if (this.coordinator.isScanRunning()) {
          this.stopScan();
          setTimeout(() => this.startScan(), 150);
        }
      });
    });

    this.btnToggleTorch.addEventListener('click', () => this.toggleTorch());

    this.btnToggleVoice.addEventListener('click', () => {
      this.isVoiceEnabled = !this.isVoiceEnabled;
      this.btnToggleVoice.classList.toggle('active', this.isVoiceEnabled);
      this.btnToggleVoice.textContent = this.isVoiceEnabled ? 'Active' : 'Muted';
      if (!this.isVoiceEnabled) {
        stopSpeaking();
      }
    });

    this.btnRequestPermission.addEventListener('click', () => this.requestCameraPermissionCheck());

    // Clinical Drawer elements
    this.companionDrawer = document.getElementById('companion-drawer') as HTMLElement;
    this.severityPill = document.getElementById('severity-pill') as HTMLElement;
    this.companionSummaryEl = document.getElementById('companion-summary') as HTMLElement;
    this.companionAdviceList = document.getElementById('companion-advice-list') as HTMLElement;
    this.btnSpeakGuidance = document.getElementById('btn-speak-guidance') as HTMLButtonElement;
    this.closeDrawerBtn = document.getElementById('close-drawer') as HTMLButtonElement;

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

    // Canvas resize listener
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    window.addEventListener('beforeunload', () => {
      if (this.waveformAnimId !== null) {
        cancelAnimationFrame(this.waveformAnimId);
      }
      this.coordinator.stopProcessing();
      stopSpeaking();
    });
  }

  private resizeCanvas(): void {
    if (!this.waveformCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.waveformCanvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      this.waveformCanvas.width = rect.width * dpr;
      this.waveformCanvas.height = rect.height * dpr;
    }
  }

  public switchTab(tab: AppTab): void {
    (Object.keys(this.tabViews) as AppTab[]).forEach((t) => {
      this.tabViews[t].classList.toggle('active', t === tab);
      this.navTabBtns[t].classList.toggle('active', t === tab);
    });

    if (tab === 'scanner') {
      requestAnimationFrame(() => this.resizeCanvas());
    } else if (tab === 'history') {
      this.refreshHistory();
    } else if (tab === 'settings') {
      this.refreshDiagnostics();
    }
  }

  private selectMode(mode: ScanMode, autoRestart: boolean = false): void {
    const wasRunning = this.coordinator.isScanRunning();
    if (wasRunning) {
      this.stopScan();
    }

    this.currentMode = mode;
    this.setIdleStateDisplay();

    const isFace = mode === 'face';
    this.videoEl.classList.toggle('finger-mode', !isFace);
    this.faceTargetBox.style.display = isFace ? 'block' : 'none';
    this.fingerTargetBox.style.display = isFace ? 'none' : 'flex';
    this.waveformModeLabel.textContent = isFace ? 'Face rPPG' : 'Finger Capillary';

    this.modeCardFace.classList.toggle('active', isFace);
    this.modeCardFinger.classList.toggle('active', !isFace);
    this.badgeModeFace.textContent = isFace ? 'Active' : 'Standby';
    this.badgeModeFinger.textContent = isFace ? 'Standby' : 'Active';

    this.cameraStatusText.textContent = isFace
      ? 'Face Mode — Tap Initiate Scan'
      : 'Finger Mode — Tap Initiate Scan';

    if (autoRestart || wasRunning) {
      this.switchTab('scanner');
      setTimeout(() => this.startScan(), 150);
    }
  }

  private async toggleTorch(): Promise<void> {
    this.isTorchPreferred = !this.isTorchPreferred;
    const camMgr = this.coordinator.getCameraManager();
    let actualState = this.isTorchPreferred;

    if (camMgr.getStream()) {
      const ok = await camMgr.setTorch(this.isTorchPreferred);
      actualState = camMgr.isTorchOn();
      if (!ok && this.isTorchPreferred) {
        this.modeTorchStatus.textContent = 'Torch hardware unavailable on active lens';
      }
    }

    this.btnModeTorch.classList.toggle('active', actualState);
    this.btnModeTorch.textContent = actualState ? 'Torch: On' : 'Torch: Off';
    this.btnToggleTorch.classList.toggle('active', actualState);
    this.btnToggleTorch.textContent = actualState ? 'On' : 'Off';
    this.refreshDiagnostics();
  }

  private initCoordinatorCallbacks(): void {
    this.coordinator.setCallbacks({
      onStateChange: (state: TriageState) => this.handleStateChange(state),
      onWaveformSample: (sample: number) => this.handleWaveformSample(sample),
      onQualityDiagnostic: (diag: QualityDiagnostic) => this.handleQualityDiagnostic(diag),
      onLiveTelemetry: (snap: LiveTelemetrySnapshot) => this.handleLiveTelemetry(snap),
      onScanComplete: (telemetry: VitalsTelemetry) => this.handleScanComplete(telemetry),
      onError: (err: Error) => this.handleError(err),
    });
  }

  private async startScan(): Promise<void> {
    if (this.isStartingScan || this.coordinator.isScanRunning()) {
      return;
    }

    this.isStartingScan = true;
    this.clearQualityDropTimer();
    this.isCalculationFrozen = false;
    this.hasActiveSignalSamples = false;
    this.waveformPoints.length = 0;
    this.btnOpenInsights.style.display = 'none';
    this.companionDrawer.style.display = 'none';
    this.progressBar.style.width = '0%';

    this.btnPrimaryScan.textContent = 'Stop Scan';
    this.btnPrimaryScan.classList.add('scanning');
    this.cameraPillFrame.classList.add('scanning');
    this.scanStartPerfTime = performance.now();
    this.lastFpsUpdatePerf = performance.now();
    this.frameSampleCount = 0;

    try {
      await this.coordinator.startScan(this.currentMode, this.videoEl);
      this.refreshDiagnostics();
    } catch (err: any) {
      this.handleError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      this.isStartingScan = false;
    }
  }

  private stopScan(): void {
    this.clearQualityDropTimer();
    this.coordinator.stopProcessing();
    this.hasActiveSignalSamples = false;
    this.isCalculationFrozen = false;
    this.waveformPoints.length = 0;
    this.setIdleStateDisplay();
  }

  private clearQualityDropTimer(): void {
    if (this.qualityDropTimer !== null) {
      clearTimeout(this.qualityDropTimer);
      this.qualityDropTimer = null;
    }
  }

  /**
   * Strict Zero-Mock Default State:
   * Every card returns to N/A and waveform flattens to baseline.
   */
  private setIdleStateDisplay(): void {
    this.hrEl.textContent = 'N/A';
    this.hrUnitEl.textContent = '';
    this.rrEl.textContent = 'N/A';
    this.rrUnitEl.textContent = '';
    this.sqiEl.textContent = 'N/A';
    this.sqiUnitEl.textContent = '';
    this.asymmetryEl.textContent = 'N/A';
    this.asymmetryEl.style.color = '';
    this.asymmetrySubEl.textContent = 'Awaiting scan';
    this.rhythmEl.textContent = 'N/A';
    this.rhythmSubEl.textContent = 'IBI regularity';
    this.stressEl.textContent = 'N/A';
    this.stressSubEl.textContent = 'HRV RMSSD';
    this.pulseIconEl?.classList.remove('heart-pulse');

    this.qualityTextEl.textContent = 'Signal: Idle';
    this.statusDotEl.className = 'status-dot idle';
    this.cameraStatusText.textContent = 'Idle — Tap Initiate Scan';
    this.progressBar.style.width = '0%';
    this.btnPrimaryScan.textContent = 'Initiate Scan';
    this.btnPrimaryScan.classList.remove('scanning');
    this.cameraPillFrame.classList.remove('scanning');
    this.measuredFps = 0;
    if (this.diagFpsEl) this.diagFpsEl.textContent = '-- FPS';
  }

  private setCalibratingDisplay(): void {
    this.hrEl.textContent = 'Calibrating...';
    this.hrEl.className = 'card-status-text';
    this.hrUnitEl.textContent = '';

    this.rrEl.textContent = 'Calibrating...';
    this.rrEl.className = 'card-status-text';
    this.rrUnitEl.textContent = '';

    this.sqiEl.textContent = 'Calibrating...';
    this.sqiEl.className = 'card-status-text';
    this.sqiUnitEl.textContent = '';

    this.asymmetryEl.textContent = 'Calibrating...';
    this.asymmetryEl.style.color = '';
    this.asymmetrySubEl.textContent = this.currentMode === 'face' ? 'Detecting face landmarks' : 'Face mode only';

    this.rhythmEl.textContent = 'Calibrating...';
    this.rhythmSubEl.textContent = 'Sampling peaks';

    this.stressEl.textContent = 'Calibrating...';
    this.stressSubEl.textContent = 'Measuring HRV';
  }

  private resetMetricsOnlyToNA(): void {
    this.hrEl.textContent = 'N/A';
    this.hrEl.className = 'card-big-num';
    this.hrUnitEl.textContent = '';

    this.rrEl.textContent = 'N/A';
    this.rrEl.className = 'card-big-num';
    this.rrUnitEl.textContent = '';

    this.sqiEl.textContent = 'N/A';
    this.sqiEl.className = 'card-big-num';
    this.sqiUnitEl.textContent = '';

    this.asymmetryEl.textContent = 'N/A';
    this.asymmetryEl.style.color = '';
    this.asymmetrySubEl.textContent = 'Signal lost';

    this.rhythmEl.textContent = 'N/A';
    this.rhythmSubEl.textContent = 'Signal lost';

    this.stressEl.textContent = 'N/A';
    this.stressSubEl.textContent = 'Signal lost';
    this.pulseIconEl?.classList.remove('heart-pulse');
  }

  private handleStateChange(state: TriageState): void {
    switch (state) {
      case 'IDLE':
        this.setIdleStateDisplay();
        break;
      case 'STARTING':
        this.qualityTextEl.textContent = 'Signal: Starting Camera';
        this.statusDotEl.className = 'status-dot warning';
        this.cameraStatusText.textContent = 'Starting Camera...';
        this.setCalibratingDisplay();
        break;
      case 'CALIBRATING':
        this.qualityTextEl.textContent = 'Signal: Calibrating';
        this.statusDotEl.className = 'status-dot warning';
        this.cameraStatusText.textContent = 'Calibrating Optical Baseline...';
        this.setCalibratingDisplay();
        break;
      case 'ACQUIRING':
        this.qualityTextEl.textContent = 'Signal: Acquiring';
        this.statusDotEl.className = 'status-dot';
        this.scanStartPerfTime = performance.now();
        this.pulseIconEl?.classList.add('heart-pulse');
        break;
      case 'ANALYZING':
        this.qualityTextEl.textContent = 'Signal: Finalizing';
        this.cameraStatusText.textContent = 'Computing Final Telemetry...';
        this.progressBar.style.width = '100%';
        break;
      case 'COMPLETE':
        this.qualityTextEl.textContent = 'Signal Quality: Verified';
        this.statusDotEl.className = 'status-dot';
        this.cameraStatusText.textContent = 'Scan Complete';
        this.btnPrimaryScan.textContent = 'Initiate Scan';
        this.btnPrimaryScan.classList.remove('scanning');
        this.cameraPillFrame.classList.remove('scanning');
        this.pulseIconEl?.classList.remove('heart-pulse');
        this.hasActiveSignalSamples = false;
        break;
      case 'FAILED':
        this.qualityTextEl.textContent = 'Poor Signal / Adjust Position';
        this.statusDotEl.className = 'status-dot critical';
        this.cameraStatusText.textContent = 'Scan Interrupted — Tap Initiate Scan';
        this.btnPrimaryScan.textContent = 'Initiate Scan';
        this.btnPrimaryScan.classList.remove('scanning');
        this.cameraPillFrame.classList.remove('scanning');
        this.resetMetricsOnlyToNA();
        this.hasActiveSignalSamples = false;
        this.waveformPoints.length = 0;
        break;
    }
  }

  private handleWaveformSample(sample: number): void {
    this.hasActiveSignalSamples = true;
    this.waveformPoints.push(sample);
    if (this.waveformPoints.length > 180) {
      this.waveformPoints.shift();
    }

    // Track real FPS
    this.frameSampleCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdatePerf >= 1000) {
      this.measuredFps = Math.round((this.frameSampleCount * 1000) / (now - this.lastFpsUpdatePerf));
      this.frameSampleCount = 0;
      this.lastFpsUpdatePerf = now;
      if (this.diagFpsEl) {
        this.diagFpsEl.textContent = `${this.measuredFps} FPS`;
      }
    }

    if (this.coordinator.getState() === 'ACQUIRING') {
      const elapsed = now - this.scanStartPerfTime;
      const remainingSec = Math.max(1, Math.ceil((15000 - elapsed) / 1000));
      if (!this.isCalculationFrozen) {
        this.cameraStatusText.textContent = `Hold Steady · ${remainingSec}s remaining`;
      }
      const pct = Math.min(100, Math.max(0, (elapsed / 15000) * 100));
      this.progressBar.style.width = `${pct}%`;
    }
  }

  private handleQualityDiagnostic(diag: QualityDiagnostic): void {
    const state = this.coordinator.getState();

    if (diag.isAcceptable) {
      this.clearQualityDropTimer();
      this.isCalculationFrozen = false;
      this.statusDotEl.className = 'status-dot';
      if (state === 'CALIBRATING') {
        this.qualityTextEl.textContent = 'Signal: Calibrating';
      } else if (state === 'ACQUIRING') {
        this.qualityTextEl.textContent = 'Signal: Acceptable';
      }
      return;
    }

    // Signal drop or invalid condition -> freeze active calculation immediately
    this.isCalculationFrozen = true;
    this.statusDotEl.className = 'status-dot warning';
    this.qualityTextEl.textContent = 'Poor Signal / Adjust Position';

    if (diag.fingerContactLost) {
      this.cameraStatusText.textContent = 'Place Finger Firmly Over Rear Lens';
    } else if (diag.insufficientLuminance) {
      this.cameraStatusText.textContent = 'Low Light — Move to Brighter Area';
    } else if (diag.motionArtifactDetected) {
      this.cameraStatusText.textContent = 'Center Face & Hold Still';
    } else {
      this.cameraStatusText.textContent = 'Poor Signal — Hold Steady';
    }

    // After 1.5s continuous poor quality, return active metrics to N/A
    if (this.qualityDropTimer === null && (state === 'ACQUIRING' || state === 'CALIBRATING')) {
      this.qualityDropTimer = setTimeout(() => {
        if (this.isCalculationFrozen) {
          this.resetMetricsOnlyToNA();
          this.hasActiveSignalSamples = false;
          this.waveformPoints.length = 0;
        }
        this.qualityDropTimer = null;
      }, 1500);
    }
  }

  private handleLiveTelemetry(snap: LiveTelemetrySnapshot): void {
    if (this.isCalculationFrozen || !snap.isQualityAcceptable) {
      return;
    }

    if (snap.sqiScore !== null) {
      const sqiPct = Math.round(snap.sqiScore * 100);
      this.qualityTextEl.textContent = `Signal Quality: ${sqiPct}%`;
      this.sqiEl.className = 'card-big-num';
      this.sqiEl.textContent = `${sqiPct}%`;
      this.sqiUnitEl.textContent = 'SQI';
    } else {
      this.sqiEl.className = 'card-status-text';
      this.sqiEl.textContent = 'Gathering...';
      this.sqiUnitEl.textContent = '';
    }

    if (snap.heartRateBpm !== null && snap.heartRateBpm > 0) {
      this.hrEl.className = 'card-big-num';
      this.hrEl.textContent = Math.round(snap.heartRateBpm).toString();
      this.hrUnitEl.textContent = 'BPM';
    } else {
      this.hrEl.className = 'card-status-text';
      this.hrEl.textContent = 'Gathering...';
      this.hrUnitEl.textContent = '';
    }

    if (snap.respirationRateRpm !== null && snap.respirationRateRpm > 0) {
      this.rrEl.className = 'card-big-num';
      this.rrEl.textContent = Math.round(snap.respirationRateRpm).toString();
      this.rrUnitEl.textContent = 'RPM';
    } else {
      this.rrEl.className = 'card-status-text';
      this.rrEl.textContent = 'Gathering...';
      this.rrUnitEl.textContent = '';
    }

    if (this.currentMode === 'finger') {
      this.asymmetryEl.textContent = 'N/A';
      this.asymmetryEl.style.color = '';
      this.asymmetrySubEl.textContent = 'Face mode only';
    } else if (snap.strokeAsymmetryDelta !== null) {
      const isAsym = snap.strokeAsymmetryDelta >= 0.12;
      this.asymmetryEl.textContent = isAsym ? 'Asymmetric' : 'Normal';
      this.asymmetryEl.style.color = isAsym ? '#ff453a' : '#34c759';
      this.asymmetrySubEl.textContent = `Delta ${snap.strokeAsymmetryDelta.toFixed(3)}`;
    } else {
      this.asymmetryEl.textContent = 'Tracking...';
      this.asymmetryEl.style.color = '';
      this.asymmetrySubEl.textContent = 'Aligning landmarks';
    }

    if (snap.pulseRhythm) {
      this.rhythmEl.textContent = snap.pulseRhythm;
      this.rhythmSubEl.textContent = 'Computed from IBI';
    } else {
      this.rhythmEl.textContent = 'Gathering...';
      this.rhythmSubEl.textContent = 'Need 3+ beats';
    }

    if (snap.stressLevel) {
      this.stressEl.textContent = snap.stressLevel;
      this.stressSubEl.textContent = snap.hrvRmssdMs !== null ? `RMSSD ${snap.hrvRmssdMs} ms` : 'HRV active';
    } else {
      this.stressEl.textContent = 'Gathering...';
      this.stressSubEl.textContent = 'Need 3+ beats';
    }
  }

  private async handleScanComplete(telemetry: VitalsTelemetry): Promise<void> {
    this.clearQualityDropTimer();
    this.isCalculationFrozen = false;

    const hrValid = telemetry.heartRateBpm > 0;
    this.hrEl.className = 'card-big-num';
    this.hrEl.textContent = hrValid ? Math.round(telemetry.heartRateBpm).toString() : 'N/A';
    this.hrUnitEl.textContent = hrValid ? 'BPM' : '';

    this.rrEl.className = 'card-big-num';
    this.rrEl.textContent =
      telemetry.respirationRateRpm > 0 ? Math.round(telemetry.respirationRateRpm).toString() : 'N/A';
    this.rrUnitEl.textContent = telemetry.respirationRateRpm > 0 ? 'RPM' : '';

    const sqiPct = Math.round(telemetry.sqiScore * 100);
    this.sqiEl.className = 'card-big-num';
    this.sqiEl.textContent = `${sqiPct}%`;
    this.sqiUnitEl.textContent = 'SQI';
    this.qualityTextEl.textContent = `Signal Quality: ${sqiPct}%`;

    if (this.currentMode === 'finger' || !telemetry.hasFacialLandmarks) {
      this.asymmetryEl.textContent = this.currentMode === 'finger' ? 'N/A' : 'Optical Only';
      this.asymmetryEl.style.color = '';
      this.asymmetrySubEl.textContent =
        this.currentMode === 'finger' ? 'Face mode only' : 'No 3D landmarks';
    } else {
      const isAsymmetric = telemetry.strokeAsymmetryDelta >= 0.12;
      this.asymmetryEl.textContent = isAsymmetric ? 'Asymmetric' : 'Normal';
      this.asymmetryEl.style.color = isAsymmetric ? '#ff453a' : '#34c759';
      this.asymmetrySubEl.textContent = `Delta ${telemetry.strokeAsymmetryDelta.toFixed(3)}`;
    }

    this.rhythmEl.textContent = telemetry.pulseRhythm ?? 'N/A';
    this.rhythmSubEl.textContent = telemetry.pulseRhythm && telemetry.pulseRhythm !== 'N/A'
      ? 'Median IBI variance'
      : 'Insufficient peaks';

    this.stressEl.textContent = telemetry.stressLevel ?? 'N/A';
    this.stressSubEl.textContent = telemetry.hrvRmssdMs !== undefined
      ? `RMSSD ${telemetry.hrvRmssdMs} ms`
      : 'Insufficient IBI';

    // Save to local store and inspect baseline
    const previous = getLatestSession();
    saveSession({ mode: this.currentMode, telemetry });
    this.refreshHistory();
    this.refreshDiagnostics();

    const baseline = previous ? compareWithBaseline(telemetry, previous) : undefined;
    const guidance = await getTriageGuidance({
      telemetry,
      mode: this.currentMode,
      baselineDelta: baseline,
    });

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
    this.btnOpenInsights.style.display = 'inline-flex';
    this.companionDrawer.style.display = 'block';

    if (this.isVoiceEnabled) {
      speakGuidance(this.latestGuidanceSpeechText);
    }
  }

  private handleError(err: Error): void {
    this.clearQualityDropTimer();
    this.isCalculationFrozen = true;
    this.hasActiveSignalSamples = false;
    this.waveformPoints.length = 0;

    this.cameraStatusText.textContent = err.message || 'Sensor Error';
    this.qualityTextEl.textContent = 'Poor Signal / Error';
    this.statusDotEl.className = 'status-dot critical';
    this.btnPrimaryScan.textContent = 'Initiate Scan';
    this.btnPrimaryScan.classList.remove('scanning');
    this.cameraPillFrame.classList.remove('scanning');
    this.resetMetricsOnlyToNA();
  }

  private refreshHistory(): void {
    if (!this.historyListEl) return;
    const sessions = listSessions(50);
    this.historyListEl.innerHTML = '';

    if (sessions.length === 0) {
      this.historyListEl.innerHTML = `
        <div class="liquid-glass-card history-item" style="text-align:center; color:var(--text-tertiary); padding:24px;">
          No recorded sessions yet. Complete a 15-second scan to log real baseline telemetry.
        </div>
      `;
      return;
    }

    sessions.forEach((s) => {
      const card = document.createElement('div');
      card.className = 'liquid-glass-card history-item';
      const dateStr = new Date(s.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
      const timeStr = new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const modeLabel = s.mode === 'face' ? 'Face rPPG' : 'Finger Capillary';
      const hrStr = s.telemetry.heartRateBpm > 0 ? `${Math.round(s.telemetry.heartRateBpm)} BPM` : 'N/A';
      const rrStr = s.telemetry.respirationRateRpm > 0 ? `${Math.round(s.telemetry.respirationRateRpm)} RPM` : 'N/A';
      const sqiStr = `${Math.round(s.telemetry.sqiScore * 100)}% SQI`;
      const rhythmStr = s.telemetry.pulseRhythm ?? 'N/A';
      const stressStr = s.telemetry.stressLevel ?? 'N/A';

      card.innerHTML = `
        <div class="history-item-top">
          <span class="history-item-title">${modeLabel} · ${hrStr}</span>
          <span class="history-item-sqi">${sqiStr}</span>
        </div>
        <div class="history-item-meta">
          ${dateStr} at ${timeStr} · Breathing: ${rrStr} · Rhythm: ${rhythmStr} · Stress: ${stressStr}
        </div>
      `;
      this.historyListEl.appendChild(card);
    });
  }

  private exportHistoryJson(): void {
    const sessions = listSessions();
    const blob = new Blob([JSON.stringify(sessions, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `triagelens-history-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  private async refreshDiagnostics(): Promise<void> {
    if (!this.diagCameraResEl) return;

    const camDiag = this.coordinator.getCameraManager().getDiagnostics();
    if (camDiag.active) {
      this.diagCameraResEl.textContent = `${camDiag.width}×${camDiag.height} (${camDiag.facingMode})`;
      this.diagTorchSupportEl.textContent = camDiag.torchSupported
        ? camDiag.torchOn ? 'Supported (On)' : 'Supported (Off)'
        : 'Not Supported on Active Lens';
    } else {
      const pref = this.coordinator.getCameraResolution();
      this.diagCameraResEl.textContent = `Idle (${pref.width}×${pref.height} configured)`;
      this.diagTorchSupportEl.textContent = 'Ready (Checks on active stream)';
    }

    this.diagFpsEl.textContent = this.measuredFps > 0 ? `${this.measuredFps} FPS` : '-- FPS';

    const faceStatus = this.coordinator.getFaceEngineStatus();
    if (faceStatus.mediaPipeActive) {
      this.diagOfflineModelsEl.textContent = '468-pt MediaPipe Mesh Active';
    } else if (faceStatus.fallbackActive) {
      this.diagOfflineModelsEl.textContent = 'Optical Forehead ROI Active';
    } else {
      this.diagOfflineModelsEl.textContent = 'Local FaceMesh WASM Ready';
    }

    try {
      if (navigator.permissions && navigator.permissions.query) {
        const res = await navigator.permissions.query({ name: 'camera' as PermissionName });
        this.diagPermissionEl.textContent = res.state.charAt(0).toUpperCase() + res.state.slice(1);
      } else {
        this.diagPermissionEl.textContent = 'Managed by OS';
      }
    } catch {
      this.diagPermissionEl.textContent = 'Managed by OS';
    }
  }

  private async requestCameraPermissionCheck(): Promise<void> {
    this.btnRequestPermission.textContent = 'Checking...';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stream.getTracks().forEach((t) => t.stop());
      this.diagPermissionEl.textContent = 'Granted';
      this.btnRequestPermission.textContent = 'Granted';
    } catch (err: any) {
      this.diagPermissionEl.textContent = err?.name === 'NotAllowedError' ? 'Denied' : 'Unavailable';
      this.btnRequestPermission.textContent = 'Retry Access';
    }
    setTimeout(() => {
      this.btnRequestPermission.textContent = 'Check Access';
    }, 2000);
  }

  /**
   * Waveform Renderer (Zero Mock Data):
   *   - Idle / No Valid Stream: Flat zero baseline + subtle idle sweep indicator.
   *   - Active Valid Signal: Plots genuine filteredWaveformSample buffer with AGC.
   */
  private startWaveformLoop(): void {
    let sweepProgress = 0;
    let rollingMaxAmp = 0.02;

    const draw = () => {
      if (this.waveformCtx && this.waveformCanvas) {
        const w = this.waveformCanvas.width;
        const h = this.waveformCanvas.height;

        if (w > 0 && h > 0) {
          const dpr = window.devicePixelRatio || 1;
          this.waveformCtx.clearRect(0, 0, w, h);

          // Subtle oscilloscope grid
          this.waveformCtx.strokeStyle = 'rgba(255, 255, 255, 0.045)';
          this.waveformCtx.lineWidth = 1 * dpr;
          const gridSpacing = 20 * dpr;
          this.waveformCtx.beginPath();
          for (let gx = 0; gx <= w; gx += gridSpacing) {
            this.waveformCtx.moveTo(gx, 0);
            this.waveformCtx.lineTo(gx, h);
          }
          for (let gy = 0; gy <= h; gy += gridSpacing) {
            this.waveformCtx.moveTo(0, gy);
            this.waveformCtx.lineTo(w, gy);
          }
          this.waveformCtx.stroke();

          const centerY = h / 2;
          const isScanning = this.coordinator.isScanRunning();
          const points = this.waveformPoints;

          if (isScanning && this.hasActiveSignalSamples && !this.isCalculationFrozen && points.length >= 10) {
            // Plot REAL filtered capillary PPG samples strictly from DSP
            let localMax = 0.001;
            const tailStart = Math.max(0, points.length - 60);
            for (let k = tailStart; k < points.length; k++) {
              const absVal = Math.abs(points[k]);
              if (absVal > localMax) localMax = absVal;
            }
            rollingMaxAmp = rollingMaxAmp * 0.94 + localMax * 0.06;
            const maxH = h * 0.38;
            const normScale = maxH / Math.max(0.006, rollingMaxAmp);

            const displayCount = Math.min(points.length, 90);
            const startIndex = points.length - displayCount;
            const stepX = w / (displayCount - 1);

            this.waveformCtx.strokeStyle = '#ffffff';
            this.waveformCtx.lineWidth = 2.0 * dpr;
            this.waveformCtx.shadowColor = 'rgba(255, 255, 255, 0.7)';
            this.waveformCtx.shadowBlur = 5 * dpr;
            this.waveformCtx.lineCap = 'round';
            this.waveformCtx.lineJoin = 'round';
            this.waveformCtx.beginPath();

            let prevX = 0;
            let prevY = centerY - Math.max(-maxH, Math.min(maxH, points[startIndex] * normScale));
            this.waveformCtx.moveTo(prevX, prevY);

            for (let idx = 1; idx < displayCount; idx++) {
              const currX = idx * stepX;
              const val = points[startIndex + idx] || 0;
              const currY = centerY - Math.max(-maxH, Math.min(maxH, val * normScale));
              const midX = (prevX + currX) / 2;
              const midY = (prevY + currY) / 2;
              this.waveformCtx.quadraticCurveTo(prevX, prevY, midX, midY);
              prevX = currX;
              prevY = currY;
            }
            this.waveformCtx.lineTo(prevX, prevY);
            this.waveformCtx.stroke();
            this.waveformCtx.shadowBlur = 0;
          } else {
            // Flat zero baseline with a subtle idle sweep line (NO fake heartbeat!)
            this.waveformCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
            this.waveformCtx.lineWidth = 1.4 * dpr;
            this.waveformCtx.beginPath();
            this.waveformCtx.moveTo(0, centerY);
            this.waveformCtx.lineTo(w, centerY);
            this.waveformCtx.stroke();

            // Subtle sweeping cursor along the zero baseline
            sweepProgress = (sweepProgress + 0.006) % 1;
            const sweepX = sweepProgress * w;
            const grad = this.waveformCtx.createLinearGradient(
              Math.max(0, sweepX - 48 * dpr),
              centerY,
              sweepX,
              centerY
            );
            grad.addColorStop(0, 'rgba(255, 255, 255, 0)');
            grad.addColorStop(1, 'rgba(255, 255, 255, 0.75)');

            this.waveformCtx.strokeStyle = grad;
            this.waveformCtx.lineWidth = 2.0 * dpr;
            this.waveformCtx.beginPath();
            this.waveformCtx.moveTo(Math.max(0, sweepX - 48 * dpr), centerY);
            this.waveformCtx.lineTo(sweepX, centerY);
            this.waveformCtx.stroke();
          }
        }
      }

      this.waveformAnimId = requestAnimationFrame(draw);
    };

    draw();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new TriageAppUI();
});
