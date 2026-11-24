/**
 * TriageLens Local Session Storage & Baseline Analysis
 * 
 * Offline-first session history persistence with safe automatic in-memory fallback.
 * Strictly stores verified numerical telemetry without raw video frames or PII.
 */

import {
  ScanMode,
  TriageSessionRecord,
  VitalsBaselineDelta,
  VitalsTelemetry,
} from '../types/triage';

export const TRIAGE_STORAGE_KEY = 'triagelens_sessions_v1';
export const MAX_STORED_SESSIONS = 100;

class MemoryStorageFallback {
  private store = new Map<string, string>();

  public getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  public setItem(key: string, value: string): void {
    this.store.set(key, value);
  }

  public removeItem(key: string): void {
    this.store.delete(key);
  }

  public clear(): void {
    this.store.clear();
  }
}

/**
 * Access safe storage mechanism (localStorage with automatic memory fallback).
 */
function getStorage(): { getItem(k: string): string | null; setItem(k: string, v: string): void; removeItem(k: string): void } {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      // Test storage capability (handles Safari private mode quota restrictions)
      const testKey = '__triage_storage_test__';
      window.localStorage.setItem(testKey, '1');
      window.localStorage.removeItem(testKey);
      return window.localStorage;
    }
  } catch {
    // Fall back to memory
  }
  return inMemoryStore;
}

const inMemoryStore = new MemoryStorageFallback();

/**
 * Generate a unique monotonic session ID.
 */
function generateSessionId(): string {
  const ts = Date.now();
  const rand = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
  return `session_${ts}_${rand}`;
}

/**
 * Save a completed vital telemetry scan to local storage.
 */
export function saveSession(
  input: {
    mode: ScanMode;
    telemetry: VitalsTelemetry;
    durationMs?: number;
    qualitySummary?: {
      averageSqi: number;
      averageLuma: number;
      acceptableRate: number;
    };
    notes?: string;
  }
): TriageSessionRecord {
  const storage = getStorage();
  const timestamp = input.telemetry.timestamp || Date.now();
  const record: TriageSessionRecord = {
    id: generateSessionId(),
    timestamp,
    mode: input.mode,
    telemetry: input.telemetry,
    durationMs: input.durationMs ?? 15000,
    qualitySummary: input.qualitySummary ?? {
      averageSqi: input.telemetry.sqiScore,
      averageLuma: 128,
      acceptableRate: 1.0,
    },
    notes: input.notes,
  };

  const existing = listSessions();
  const updated = [record, ...existing].slice(0, MAX_STORED_SESSIONS);

  try {
    storage.setItem(TRIAGE_STORAGE_KEY, JSON.stringify(updated));
  } catch {
    // If quota exceeded, trim older sessions and retry
    const trimmed = [record, ...existing.slice(0, 20)];
    try {
      storage.setItem(TRIAGE_STORAGE_KEY, JSON.stringify(trimmed));
    } catch {
      inMemoryStore.setItem(TRIAGE_STORAGE_KEY, JSON.stringify(updated));
    }
  }

  return record;
}

/**
 * List all saved sessions sorted newest first.
 */
export function listSessions(limit?: number): TriageSessionRecord[] {
  const storage = getStorage();
  const raw = storage.getItem(TRIAGE_STORAGE_KEY);
  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const valid = parsed.filter(
      (s): s is TriageSessionRecord => s && typeof s.id === 'string' && s.telemetry != null
    );
    return limit && limit > 0 ? valid.slice(0, limit) : valid;
  } catch {
    return [];
  }
}

/**
 * Retrieve a specific session by ID.
 */
export function getSession(id: string): TriageSessionRecord | null {
  const sessions = listSessions();
  return sessions.find((s) => s.id === id) ?? null;
}

/**
 * Retrieve the most recent completed scan session.
 */
export function getLatestSession(): TriageSessionRecord | null {
  const sessions = listSessions(1);
  return sessions.length > 0 ? sessions[0] : null;
}

/**
 * Delete a specific session by ID.
 */
export function deleteSession(id: string): boolean {
  const storage = getStorage();
  const sessions = listSessions();
  const filtered = sessions.filter((s) => s.id !== id);
  if (filtered.length === sessions.length) return false;

  storage.setItem(TRIAGE_STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Clear all stored triage session history.
 */
export function clearSessions(): void {
  const storage = getStorage();
  storage.removeItem(TRIAGE_STORAGE_KEY);
  inMemoryStore.clear();
}

/**
 * Compare an active vitals reading against a prior baseline session or reading.
 */
export function compareWithBaseline(
  current: VitalsTelemetry,
  baseline?: VitalsTelemetry | TriageSessionRecord | null
): VitalsBaselineDelta {
  const baselineTelemetry: VitalsTelemetry | null =
    baseline == null
      ? null
      : 'telemetry' in baseline
      ? baseline.telemetry
      : baseline;

  if (!baselineTelemetry) {
    return {
      deltaHeartRateBpm: 0,
      deltaRespirationRateRpm: 0,
      deltaStrokeAsymmetry: 0,
      timeSinceBaselineMs: 0,
      hasElevatedHeartRate: current.heartRateBpm > 100,
      hasElevatedRespiration: current.respirationRateRpm > 24,
      hasAsymmetryAlert: current.strokeAsymmetryDelta >= 0.12,
    };
  }

  const deltaHeartRateBpm = current.heartRateBpm - baselineTelemetry.heartRateBpm;
  const deltaRespirationRateRpm = current.respirationRateRpm - baselineTelemetry.respirationRateRpm;
  const deltaStrokeAsymmetry = current.strokeAsymmetryDelta - baselineTelemetry.strokeAsymmetryDelta;
  const timeSinceBaselineMs = Math.max(0, current.timestamp - baselineTelemetry.timestamp);

  const hasElevatedHeartRate = current.heartRateBpm > 100 || deltaHeartRateBpm >= 20;
  const hasElevatedRespiration = current.respirationRateRpm > 24 || deltaRespirationRateRpm >= 6;
  const hasAsymmetryAlert = current.strokeAsymmetryDelta >= 0.12 || deltaStrokeAsymmetry >= 0.07;

  return {
    deltaHeartRateBpm: Math.round(deltaHeartRateBpm * 10) / 10,
    deltaRespirationRateRpm: Math.round(deltaRespirationRateRpm * 10) / 10,
    deltaStrokeAsymmetry: Math.round(deltaStrokeAsymmetry * 1000) / 1000,
    timeSinceBaselineMs,
    hasElevatedHeartRate,
    hasElevatedRespiration,
    hasAsymmetryAlert,
  };
}
