/**
 * TriageLens Core Engine & Non-UI Application Layer
 * 
 * Complete public API exporting:
 *   - Type definitions & contracts
 *   - Camera hardware stream manager
 *   - FaceMesh rPPG & FAST stroke geometry engine
 *   - Finger contact PPG transillumination engine
 *   - Deterministic DSP filters, POS, resampler, biquads, peak detector, & SQI
 *   - Triage scan coordinator state machine
 *   - Session history persistence & baseline analytics (triageStore)
 *   - AI triage companion guidance & offline clinical rules (triageCompanion)
 */

// Types & Contracts
export * from './types/triage';

// Core Hardware & Vision Engines
export { CameraStreamManager } from './core/CameraStreamManager';
export { FaceMeshEngine, type FaceMeshEngineConfig } from './core/FaceMeshEngine';
export { FingerEngine } from './core/FingerEngine';

// DSP Pipeline
export {
  BiquadFilter,
  BiquadCascade,
  PlaneOrthogonalToSkin,
  PeakDetector,
  SignalQualityIndex,
  UniformResampler,
  type BiquadCoefficients,
} from './core/DSPFilters';

// Coordinator State Machine
export {
  TriageCoordinator,
  type TriageCoordinatorCallbacks,
} from './core/TriageCoordinator';

// Session Persistence & Analytics
export {
  saveSession,
  listSessions,
  getSession,
  getLatestSession,
  deleteSession,
  clearSessions,
  compareWithBaseline,
  TRIAGE_STORAGE_KEY,
  MAX_STORED_SESSIONS,
} from './services/triageStore';

// AI Triage Companion & Offline Fallback
export {
  getTriageGuidance,
  generateOfflineGuidance,
  speakGuidance,
  stopSpeaking,
  type TtsSpeechOptions,
} from './services/triageCompanion';
