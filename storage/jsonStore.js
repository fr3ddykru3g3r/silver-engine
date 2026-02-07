import fs from "node:fs/promises";
import path from "node:path";
import { app } from "electron";

const dataDir = () => path.join(app.getPath("userData"), "storage");
const storePath = () => path.join(dataDir(), "subscriptions.json");
const prefsPath = () => path.join(dataDir(), "preferences.json");

const backupName = (suffix) => `subscriptions.backup.${suffix}.json`;

export const ensureStore = async () => {
  await fs.mkdir(dataDir(), { recursive: true });
  const file = storePath();
  try {
    await fs.access(file);
  } catch {
    await fs.writeFile(file, JSON.stringify({ subscriptions: [] }, null, 2));
  }

  try {
    await fs.access(prefsPath());
  } catch {
    await fs.writeFile(
      prefsPath(),
      JSON.stringify({ currency: "USD", theme: "dark", monthlyBudget: 200 }, null, 2)
    );
  }
};

const createBackup = async () => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupFile = path.join(dataDir(), backupName(timestamp));
  await fs.copyFile(storePath(), backupFile);

  const files = await fs.readdir(dataDir());
  const backups = files.filter((file) => file.startsWith("subscriptions.backup."));
  if (backups.length > 5) {
    const sorted = backups.sort();
    const toRemove = sorted.slice(0, backups.length - 5);
    await Promise.all(toRemove.map((file) => fs.unlink(path.join(dataDir(), file))));
  }
};

export const listSubscriptions = async () => {
  const content = await fs.readFile(storePath(), "utf-8");
  const data = JSON.parse(content);
  return data.subscriptions || [];
};

export const saveSubscriptions = async (subscriptions) => {
  await createBackup();
  await fs.writeFile(
    storePath(),
    JSON.stringify({ subscriptions }, null, 2)
  );
  return { saved: true };
};

export const loadPreferences = async () => {
  const content = await fs.readFile(prefsPath(), "utf-8");
  return JSON.parse(content);
};

export const savePreferences = async (prefs) => {
  await fs.writeFile(prefsPath(), JSON.stringify(prefs, null, 2));
  return { saved: true };
};

export const restoreFromBackup = async (backupFile) => {
  await fs.copyFile(backupFile, storePath());
  return { restored: true };
};
