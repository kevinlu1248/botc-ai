import fs from "node:fs";

// Must be imported before any module that reads process.env at load time.
// ES module imports are evaluated before the importing module's body, so an
// inline loader in index.js would run *after* agents.js had already read its
// model names.
try {
  const raw = fs.readFileSync(new URL("../.env", import.meta.url), "utf8");
  for (const line of raw.split("\n")) {
    if (/^\s*(#|$)/.test(line)) continue;
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!m) continue;
    const value = m[2].trim().replace(/^["']|["']$/g, "");
    if (!process.env[m[1]]) process.env[m[1]] = value;
  }
} catch {} // no .env file — fine if the keys are already in the environment
