package com.triagelens.app;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;
import androidx.core.content.FileProvider;
import java.io.File;

/**
 * JavaScript interface exposed to the WebView for seamless native app updates.
 * Provides permission check for Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
 * opens the exact system Settings screen, and downloads/installs APKs.
 */
public class AppUpdateBridge {
    private final MainActivity activity;
    private final WebView webView;

    public AppUpdateBridge(MainActivity activity, WebView webView) {
        this.activity = activity;
        this.webView = webView;
    }

    /**
     * Returns true if the app is allowed to request package installs (Android 8.0+ / API 26+)
     */
    @JavascriptInterface
    public boolean canRequestPackageInstalls() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            return activity.getPackageManager().canRequestPackageInstalls();
        }
        return true;
    }

    /**
     * Directs user to the system Settings page to toggle "Install unknown apps" for this app
     */
    @JavascriptInterface
    public void openUnknownSourcesSettings() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Intent intent = new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES);
            intent.setData(Uri.parse("package:" + activity.getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            activity.startActivity(intent);
        } else {
            Intent intent = new Intent(Settings.ACTION_SECURITY_SETTINGS);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            activity.startActivity(intent);
        }
    }

    /**
     * Returns current installed app version name and version code
     */
    @JavascriptInterface
    public String getAppVersionInfo() {
        try {
            PackageInfo pInfo = activity.getPackageManager().getPackageInfo(activity.getPackageName(), 0);
            long vCode = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P 
                ? pInfo.getLongVersionCode() 
                : pInfo.versionCode;
            return "{\"versionName\":\"" + pInfo.versionName + "\",\"versionCode\":" + vCode + "}";
        } catch (Exception e) {
            return "{\"versionName\":\"1.0.0\",\"versionCode\":1}";
        }
    }

    /**
     * Downloads an APK from apkUrl and triggers Android Package Installer via FileProvider
     */
    @JavascriptInterface
    public void downloadAndInstallApk(final String apkUrl) {
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                try {
                    startDownloadAndInstall(apkUrl);
                } catch (Exception e) {
                    dispatchJsEvent("onUpdateError", "Failed to start download: " + e.getMessage());
                }
            }
        });
    }

    private void startDownloadAndInstall(String apkUrl) {
        final Context context = activity.getApplicationContext();
        Uri uri = Uri.parse(apkUrl);

        // Delete any previously downloaded update apk to avoid collision
        final File destinationFile = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "triagelens-update.apk");
        if (destinationFile.exists()) {
            destinationFile.delete();
        }

        DownloadManager.Request request = new DownloadManager.Request(uri);
        request.setTitle("Downloading TriageLens Update");
        request.setDescription("Downloading latest APK release...");
        request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
        request.setDestinationUri(Uri.fromFile(destinationFile));
        request.setMimeType("application/vnd.android.package-archive");

        final DownloadManager manager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        if (manager == null) {
            dispatchJsEvent("onUpdateError", "DownloadManager service unavailable");
            return;
        }

        final long downloadId = manager.enqueue(request);
        dispatchJsEvent("onUpdateDownloadStarted", String.valueOf(downloadId));

        // Register receiver for when download finishes
        IntentFilter filter = new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE);
        BroadcastReceiver receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context ctx, Intent intent) {
                long id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1);
                if (id == downloadId) {
                    try {
                        ctx.unregisterReceiver(this);
                    } catch (Exception ignored) {}

                    dispatchJsEvent("onUpdateDownloadComplete", destinationFile.getAbsolutePath());
                    installApkFile(destinationFile);
                }
            }
        };

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED);
        } else {
            context.registerReceiver(receiver, filter);
        }
    }

    private void installApkFile(File apkFile) {
        try {
            if (!apkFile.exists()) {
                dispatchJsEvent("onUpdateError", "APK file missing on disk");
                return;
            }

            Intent installIntent = new Intent(Intent.ACTION_VIEW);
            installIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_GRANT_READ_URI_PERMISSION);

            Uri apkUri;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                apkUri = FileProvider.getUriForFile(
                    activity,
                    activity.getPackageName() + ".fileprovider",
                    apkFile
                );
            } else {
                apkUri = Uri.fromFile(apkFile);
            }

            installIntent.setDataAndType(apkUri, "application/vnd.android.package-archive");
            activity.startActivity(installIntent);
            dispatchJsEvent("onUpdateInstallerLaunched", "Success");
        } catch (Exception e) {
            dispatchJsEvent("onUpdateError", "Failed to launch installer: " + e.getMessage());
        }
    }

    private void dispatchJsEvent(final String eventName, final String data) {
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String safeData = data.replace("\"", "\\\"");
                String script = "window.dispatchEvent(new CustomEvent('" + eventName + "', { detail: \"" + safeData + "\" }));";
                webView.evaluateJavascript(script, null);
            }
        });
    }
}
