/**
 * TriageLens In-App Update Service
 * 
 * Handles:
 *  - Checking remote version manifest (`/version.json`)
 *  - Interfacing with native AndroidAppUpdater (via Capacitor WebView JS bridge)
 *  - Verifying Android unknown sources permission (`canRequestPackageInstalls`)
 *  - Directing user to Settings (`ACTION_MANAGE_UNKNOWN_APP_SOURCES`) if disabled
 *  - Triggering background APK download and package installer launch
 *  - Clean fallback for web/PWA browser environments
 */

import { Capacitor } from '@capacitor/core';

export interface AppVersionManifest {
  version: string;
  versionCode: number;
  releaseDate: string;
  apkUrl: string;
  minRequiredVersion?: string;
  title?: string;
  releaseNotes?: string[];
}

export interface UpdateCheckResult {
  hasUpdate: boolean;
  currentVersion: string;
  currentVersionCode: number;
  latestVersion: string;
  latestVersionCode: number;
  apkUrl: string;
  title: string;
  releaseNotes: string[];
  isNativeAndroid: boolean;
  canInstallUnknownApps: boolean;
}

declare global {
  interface Window {
    AndroidAppUpdater?: {
      canRequestPackageInstalls: () => boolean;
      openUnknownSourcesSettings: () => void;
      getAppVersionInfo: () => string; // returns JSON string { versionName, versionCode }
      downloadAndInstallApk: (apkUrl: string) => void;
    };
  }
}

export class UpdateService {
  private static readonly MANIFEST_URL = '/version.json';
  private static currentVersion = '1.0.0';
  private static currentVersionCode = 1;

  public static isNativeAndroid(): boolean {
    if (typeof window !== 'undefined' && !!window.AndroidAppUpdater) {
      return true;
    }
    try {
      return (
        (typeof Capacitor?.isNativePlatform === 'function' && Capacitor.isNativePlatform()) ||
        (typeof Capacitor?.getPlatform === 'function' && Capacitor.getPlatform() === 'android')
      );
    } catch {
      return false;
    }
  }

  public static getInstalledVersionInfo(): { versionName: string; versionCode: number } {
    if (this.isNativeAndroid() && window.AndroidAppUpdater?.getAppVersionInfo) {
      try {
        const infoStr = window.AndroidAppUpdater.getAppVersionInfo();
        const parsed = JSON.parse(infoStr);
        return {
          versionName: parsed.versionName || this.currentVersion,
          versionCode: parsed.versionCode || this.currentVersionCode,
        };
      } catch {
        // Fallback
      }
    }
    return {
      versionName: this.currentVersion,
      versionCode: this.currentVersionCode,
    };
  }

  /**
   * Checks if the app has permission to install unknown apps from this source
   */
  public static canRequestPackageInstalls(): boolean {
    if (this.isNativeAndroid() && window.AndroidAppUpdater) {
      try {
        return window.AndroidAppUpdater.canRequestPackageInstalls();
      } catch {
        return false;
      }
    }
    return true; // Web / browser doesn't have package restriction
  }

  /**
   * Directs user to the system Settings page to toggle "Install unknown apps"
   */
  public static openUnknownSourcesSettings(): void {
    if (this.isNativeAndroid() && window.AndroidAppUpdater) {
      window.AndroidAppUpdater.openUnknownSourcesSettings();
    }
  }

  /**
   * Fetches the version manifest from the server and determines if an update exists
   */
  public static async checkForUpdates(): Promise<UpdateCheckResult> {
    const installed = this.getInstalledVersionInfo();
    const isNative = this.isNativeAndroid();
    const canInstall = this.canRequestPackageInstalls();

    try {
      // Add cache buster query parameter to guarantee fresh version manifest
      const response = await fetch(`${this.MANIFEST_URL}?t=${Date.now()}`, {
        cache: 'no-store',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const manifest: AppVersionManifest = await response.json();
      const hasUpdate =
        manifest.versionCode > installed.versionCode ||
        this.compareSemver(manifest.version, installed.versionName) > 0;

      return {
        hasUpdate,
        currentVersion: installed.versionName,
        currentVersionCode: installed.versionCode,
        latestVersion: manifest.version,
        latestVersionCode: manifest.versionCode,
        apkUrl: manifest.apkUrl || '/triage-lens.apk',
        title: manifest.title || `TriageLens v${manifest.version} Available`,
        releaseNotes: manifest.releaseNotes || [
          'Performance improvements and optical telemetry fixes.',
        ],
        isNativeAndroid: isNative,
        canInstallUnknownApps: canInstall,
      };
    } catch (err) {
      return {
        hasUpdate: false,
        currentVersion: installed.versionName,
        currentVersionCode: installed.versionCode,
        latestVersion: installed.versionName,
        latestVersionCode: installed.versionCode,
        apkUrl: '/triage-lens.apk',
        title: 'Up to Date',
        releaseNotes: [],
        isNativeAndroid: isNative,
        canInstallUnknownApps: canInstall,
      };
    }
  }

  /**
   * Initiates download and installation:
   * - On Native Android: invokes native DownloadManager and FileProvider package installer
   * - On Web / Browser: triggers download link navigation
   */
  public static startUpdate(apkUrl: string): void {
    if (this.isNativeAndroid() && window.AndroidAppUpdater) {
      // Resolve full absolute URL if relative
      let fullUrl = apkUrl;
      if (fullUrl.startsWith('/')) {
        fullUrl = window.location.origin + fullUrl;
      }
      window.AndroidAppUpdater.downloadAndInstallApk(fullUrl);
    } else {
      // Browser / PWA fallback: open download in new tab or trigger location
      const link = document.createElement('a');
      link.href = apkUrl;
      link.download = 'triage-lens.apk';
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  /**
   * Simple semver comparator: 1 if vA > vB, -1 if vA < vB, 0 if equal
   */
  private static compareSemver(vA: string, vB: string): number {
    const cleanA = vA.replace(/[^0-9.]/g, '').split('.').map(n => parseInt(n, 10) || 0);
    const cleanB = vB.replace(/[^0-9.]/g, '').split('.').map(n => parseInt(n, 10) || 0);

    const length = Math.max(cleanA.length, cleanB.length);
    for (let i = 0; i < length; i++) {
      const a = cleanA[i] || 0;
      const b = cleanB[i] || 0;
      if (a > b) return 1;
      if (a < b) return -1;
    }
    return 0;
  }
}
