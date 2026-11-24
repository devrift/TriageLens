import { describe, it, expect, beforeEach } from 'vitest';
import {
  saveSession,
  listSessions,
  getSession,
  getLatestSession,
  deleteSession,
  clearSessions,
  compareWithBaseline,
} from '../triageStore';
import {
  generateOfflineGuidance,
  getTriageGuidance,
} from '../triageCompanion';
import { VitalsTelemetry } from '../../types/triage';

describe('Triage Services Test Suite', () => {
  beforeEach(() => {
    clearSessions();
  });

  const sampleTelemetry: VitalsTelemetry = {
    timestamp: 1700000000000,
    heartRateBpm: 72.0,
    respirationRateRpm: 16.0,
    strokeAsymmetryDelta: 0.02,
    sqiScore: 0.88,
    rawWaveformSample: 0.15,
    filteredWaveformSample: 0.12,
    snrDb: 12.5,
  };

  describe('triageStore: Local Session Persistence & Baseline Comparison', () => {
    it('saves a session and retrieves it in listSessions', () => {
      const record = saveSession({
        mode: 'face',
        telemetry: sampleTelemetry,
        durationMs: 15000,
      });

      expect(record.id).toBeDefined();
      expect(record.mode).toBe('face');
      expect(record.telemetry.heartRateBpm).toBe(72.0);

      const sessions = listSessions();
      expect(sessions.length).toBe(1);
      expect(sessions[0].id).toBe(record.id);
    });

    it('retrieves session by id and gets latest session', () => {
      const r1 = saveSession({
        mode: 'face',
        telemetry: { ...sampleTelemetry, heartRateBpm: 70.0 },
      });
      const r2 = saveSession({
        mode: 'finger',
        telemetry: { ...sampleTelemetry, heartRateBpm: 75.0 },
      });

      expect(getLatestSession()?.id).toBe(r2.id);
      expect(getSession(r1.id)?.telemetry.heartRateBpm).toBe(70.0);
    });

    it('deletes specific session and clears all sessions', () => {
      const r1 = saveSession({ mode: 'face', telemetry: sampleTelemetry });
      const r2 = saveSession({ mode: 'finger', telemetry: sampleTelemetry });

      expect(listSessions().length).toBe(2);
      const deleted = deleteSession(r1.id);
      expect(deleted).toBe(true);
      expect(listSessions().length).toBe(1);
      expect(listSessions()[0].id).toBe(r2.id);

      clearSessions();
      expect(listSessions().length).toBe(0);
    });

    it('compares active scan with previous baseline accurately', () => {
      const baseline: VitalsTelemetry = {
        ...sampleTelemetry,
        timestamp: 1700000000000,
        heartRateBpm: 70.0,
        respirationRateRpm: 15.0,
        strokeAsymmetryDelta: 0.02,
      };

      const currentElevated: VitalsTelemetry = {
        ...sampleTelemetry,
        timestamp: 1700000600000,
        heartRateBpm: 95.0, // +25 BPM
        respirationRateRpm: 22.0, // +7 RPM
        strokeAsymmetryDelta: 0.03,
      };

      const delta = compareWithBaseline(currentElevated, baseline);
      expect(delta.deltaHeartRateBpm).toBe(25.0);
      expect(delta.deltaRespirationRateRpm).toBe(7.0);
      expect(delta.hasElevatedHeartRate).toBe(true);
      expect(delta.hasElevatedRespiration).toBe(true);
      expect(delta.hasAsymmetryAlert).toBe(false);
      expect(delta.timeSinceBaselineMs).toBe(600000);
    });

    it('flags asymmetry alert when stroke asymmetry delta reaches alert threshold', () => {
      const baseline: VitalsTelemetry = { ...sampleTelemetry, strokeAsymmetryDelta: 0.03 };
      const currentAsymmetric: VitalsTelemetry = { ...sampleTelemetry, strokeAsymmetryDelta: 0.13 };

      const delta = compareWithBaseline(currentAsymmetric, baseline);
      expect(delta.hasAsymmetryAlert).toBe(true);
      expect(delta.deltaStrokeAsymmetry).toBe(0.10);
    });
  });

  describe('triageCompanion: Offline Clinical Guidance & Rule Engine', () => {
    it('generates NORMAL severity output on healthy resting vitals', () => {
      const output = generateOfflineGuidance({
        mode: 'face',
        telemetry: sampleTelemetry,
      });

      expect(output.severityLevel).toBe('NORMAL');
      expect(output.isOfflineFallback).toBe(true);
      expect(output.actionableAdvice.length).toBeGreaterThan(0);
      expect(output.summaryText).toContain('72 BPM');
    });

    it('generates CRITICAL alert on FAST bilateral facial asymmetry (>= 0.12)', () => {
      const output = generateOfflineGuidance({
        mode: 'face',
        telemetry: { ...sampleTelemetry, strokeAsymmetryDelta: 0.14 },
      });

      expect(output.severityLevel).toBe('CRITICAL');
      expect(output.guidanceText).toContain('FAST');
      expect(output.actionableAdvice.some((a) => a.includes('FAST check'))).toBe(true);
    });

    it('generates ELEVATED alert on tachycardia (> 100 BPM)', () => {
      const output = generateOfflineGuidance({
        mode: 'finger',
        telemetry: { ...sampleTelemetry, heartRateBpm: 115.0 },
      });

      expect(output.severityLevel).toBe('ELEVATED');
      expect(output.guidanceText).toContain('tachycardia');
    });

    it('generates INVALID alert on poor optical SQI (< 0.50)', () => {
      const output = generateOfflineGuidance({
        mode: 'face',
        telemetry: { ...sampleTelemetry, sqiScore: 0.35 },
      });

      expect(output.severityLevel).toBe('INVALID');
      expect(output.guidanceText).toContain('Optical signal-to-noise ratio was insufficient');
    });

    it('getTriageGuidance returns deterministic output without throwing when offline or unconfigured', async () => {
      const output = await getTriageGuidance({
        mode: 'face',
        telemetry: sampleTelemetry,
      });

      expect(output).toBeDefined();
      expect(output.summaryText).toBeDefined();
      expect(['NORMAL', 'ELEVATED', 'CRITICAL', 'INVALID']).toContain(output.severityLevel);
    });
  });
});
