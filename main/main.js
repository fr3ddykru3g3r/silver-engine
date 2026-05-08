import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ensureStore,
  listSubscriptions,
  saveSubscriptions,
  loadPreferences,
  savePreferences,
  restoreFromBackup,
  exportPackage
} from "../storage/jsonStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = !app.isPackaged;
const installerUrl =
  process.env.INSTALLER_URL || "https://github.com/your-org/silver-engine/releases";

const createWindow = async () => {
  await ensureStore();

  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1024,
    minHeight: 720,
    backgroundColor: "#0f1419",
    show: false,
    titleBarStyle: "hiddenInset",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.js")
    }
  });

  win.once("ready-to-show", () => win.show());

  if (isDev) {
    await win.loadURL("http://localhost:5173");
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    await win.loadFile(path.join(__dirname, "../dist/renderer/index.html"));
  }
};

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

ipcMain.handle("storage:list", async () => listSubscriptions());
ipcMain.handle("storage:save", async (_event, subscriptions) => saveSubscriptions(subscriptions));

ipcMain.handle("preferences:load", async () => loadPreferences());
ipcMain.handle("preferences:save", async (_event, prefs) => savePreferences(prefs));

ipcMain.handle("backup:restore", async () => {
  const result = await dialog.showOpenDialog({
    title: "Restore Backup",
    filters: [{ name: "JSON", extensions: ["json"] }],
    properties: ["openFile"]
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { restored: false };
  }

  return restoreFromBackup(result.filePaths[0]);
});

ipcMain.handle("backup:export", async () => {
  const result = await dialog.showSaveDialog({
    title: "Download package",
    defaultPath: "subscriptions-package.json",
    filters: [{ name: "JSON", extensions: ["json"] }]
  });

  if (result.canceled || !result.filePath) {
    return { exported: false };
  }

  return exportPackage(result.filePath);
});

ipcMain.handle("app:download-installer", async () => {
  await shell.openExternal(installerUrl);
  return { opened: true };
});
