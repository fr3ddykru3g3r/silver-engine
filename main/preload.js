import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("subscriptionAPI", {
  list: () => ipcRenderer.invoke("storage:list"),
  save: (subscriptions) => ipcRenderer.invoke("storage:save", subscriptions),
  loadPreferences: () => ipcRenderer.invoke("preferences:load"),
  savePreferences: (prefs) => ipcRenderer.invoke("preferences:save", prefs),
  restoreBackup: () => ipcRenderer.invoke("backup:restore"),
  exportPackage: () => ipcRenderer.invoke("backup:export"),
  downloadInstaller: () => ipcRenderer.invoke("app:download-installer")
});
