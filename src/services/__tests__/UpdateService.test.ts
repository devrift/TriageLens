import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UpdateService, AppVersionManifest } from '../updateService';

describe('UpdateService: In-App Update Engine', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('correctly reports installed version fallback when native bridge is absent', () => {
    const info = UpdateService.getInstalledVersionInfo();
    expect(info.versionName).toBe('1.0.0');
    expect(info.versionCode).toBe(1);
  });

  it('canRequestPackageInstalls defaults to true on web/browser environments', () => {
    const canInstall = UpdateService.canRequestPackageInstalls();
    expect(canInstall).toBe(true);
  });

  it('reads native bridge version info and permissions when AndroidAppUpdater is present', () => {
    (globalThis as any).window = {
      AndroidAppUpdater: {
        canRequestPackageInstalls: vi.fn(() => false),
        openUnknownSourcesSettings: vi.fn(),
        getAppVersionInfo: vi.fn(() => JSON.stringify({ versionName: '1.0.0', versionCode: 1 })),
        downloadAndInstallApk: vi.fn(),
      },
    };

    expect(UpdateService.canRequestPackageInstalls()).toBe(false);
    expect(UpdateService.getInstalledVersionInfo()).toEqual({
      versionName: '1.0.0',
      versionCode: 1,
    });

    UpdateService.openUnknownSourcesSettings();
    expect((window as any).AndroidAppUpdater.openUnknownSourcesSettings).toHaveBeenCalledTimes(1);

    UpdateService.startUpdate('https://tagemeds.vercel.app/triage-lens.apk');
    expect((window as any).AndroidAppUpdater.downloadAndInstallApk).toHaveBeenCalledWith(
      'https://tagemeds.vercel.app/triage-lens.apk'
    );
  });

  it('checkForUpdates detects newer version from remote manifest', async () => {
    const mockManifest: AppVersionManifest = {
      version: '1.0.1',
      versionCode: 2,
      releaseDate: '2026-11-26',
      apkUrl: '/triage-lens.apk',
      title: 'TriageLens v1.0.1 Available',
      releaseNotes: ['Unknown sources permission update handler', 'Performance enhancements'],
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockManifest,
    });

    const result = await UpdateService.checkForUpdates();
    expect(result.hasUpdate).toBe(true);
    expect(result.latestVersion).toBe('1.0.1');
    expect(result.latestVersionCode).toBe(2);
    expect(result.releaseNotes.length).toBe(2);
  });

  it('checkForUpdates reports hasUpdate: false when client is at or ahead of remote manifest', async () => {
    const mockManifest: AppVersionManifest = {
      version: '1.0.0',
      versionCode: 1,
      releaseDate: '2026-11-26',
      apkUrl: '/triage-lens.apk',
      title: 'TriageLens Up to Date',
      releaseNotes: [],
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockManifest,
    });

    const result = await UpdateService.checkForUpdates();
    expect(result.hasUpdate).toBe(false);
  });
});
