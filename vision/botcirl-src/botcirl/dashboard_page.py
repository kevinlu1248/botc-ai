"""The dashboard page. Kept as one self-contained string - no CDN, no build step.

Deliberately plain: this is an instrument for reading what the robot perceived,
so everything on it should be legible at a glance and nothing should move unless
the data moved.
"""

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>botcirl — session</title>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --line:#2a323d; --text:#e6edf3;
    --dim:#8b949e; --accent:#58a6ff; --vote:#ffb84d; --speak:#7ee787;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text",Segoe UI,sans-serif}
  header{display:flex;align-items:baseline;gap:16px;padding:12px 18px;
         border-bottom:1px solid var(--line)}
  h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
  .status{color:var(--dim);font-size:12px}
  .live{color:var(--speak)}
  .warn{color:var(--vote)}
  main{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(300px,.85fr);
       gap:14px;padding:14px;align-items:start}
  @media (max-width:900px){main{grid-template-columns:1fr}}
  .side{display:flex;flex-direction:column;gap:14px;min-width:0}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
            color:var(--dim);margin:0 0 10px}
  canvas{width:100%;display:block;border-radius:6px}
  #map{background:#0b0e13}
  #timeline{background:#0b0e13;cursor:pointer;margin-top:6px}
  .controls{display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap}
  button{background:#21262d;color:var(--text);border:1px solid var(--line);
         border-radius:6px;padding:5px 12px;font-size:13px;cursor:pointer}
  button:hover{background:#2d333b}
  button.on{background:var(--accent);color:#0b0e13;border-color:var(--accent)}
  input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}
  .clock{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12px;min-width:96px}
  ul{list-style:none;margin:0;padding:0;max-height:340px;overflow-y:auto}
  li{display:flex;gap:9px;align-items:baseline;padding:7px 8px;border-radius:6px;
     cursor:pointer;border-left:3px solid transparent}
  li:hover{background:#1c2230}
  li.active{background:#1c2230;border-left-color:var(--speak)}
  .t{color:var(--dim);font-variant-numeric:tabular-nums;font-size:12px;min-width:52px}
  .who{font-weight:600}
  .basis{color:var(--dim);font-size:11px;margin-left:auto;text-align:right}
  .swatch{width:9px;height:9px;border-radius:50%;flex:none;align-self:center}
  .empty{color:var(--dim);font-size:13px;padding:8px 2px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--dim)}
  .legend span{display:flex;align-items:center;gap:5px}
  .tab{display:inline-block;padding:3px 10px;margin-right:4px;border-radius:5px;
       cursor:pointer;color:var(--dim);background:#1c2230;text-transform:none;
       letter-spacing:0;font-size:12px}
  .tab.on{background:var(--accent);color:#0b0e13;font-weight:600}
  /* Per-session people gallery */
  #people{display:flex;flex-direction:column;gap:8px;max-height:420px;overflow-y:auto}
  .person{display:flex;gap:10px;align-items:center;padding:8px;border-radius:8px;
          background:#0b0e13;border:1px solid var(--line)}
  .person.here{border-color:color-mix(in srgb, var(--accent) 55%, var(--line))}
  .person .face{width:56px;height:56px;border-radius:50%;object-fit:cover;
                background:#1c2230;border:2px solid var(--line);flex:none}
  .person.here .face{border-color:var(--accent)}
  .person .face.placeholder{display:flex;align-items:center;justify-content:center;
                            color:var(--dim);font-size:13px;font-weight:600}
  .person .meta{min-width:0;flex:1}
  .person .name{font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis}
  .person .id{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums}
  .person .badge{font-size:10px;padding:2px 7px;border-radius:999px;flex:none;
                 border:1px solid var(--line);color:var(--dim)}
  .person.here .badge{background:rgba(126,231,135,.12);border-color:var(--speak);
                      color:var(--speak)}
  .person .badges{display:flex;flex-direction:column;gap:4px;align-items:flex-end;flex:none}
  .person .badge.looking{background:rgba(88,166,255,.15);border-color:var(--accent);
                         color:var(--accent)}
</style></head><body>
<header>
  <h1>botcirl</h1>
  <span class="status" id="status">connecting…</span>
  <span class="status" id="units"></span>
</header>
<main>
  <div class="panel">
    <h2>
      <span class="tab on" id="tabView">Camera view</span>
      <span class="tab" id="tabMap">Top-down</span>
      <span id="maptype" style="text-transform:none;letter-spacing:0"></span>
    </h2>
    <canvas id="view" width="900" height="506"></canvas>
    <canvas id="map" width="900" height="560" hidden></canvas>
    <div class="legend" id="legend"></div>
    <canvas id="timeline" width="900" height="112"></canvas>
    <div class="controls">
      <button id="play">▶ Play</button>
      <button id="liveBtn" class="on">Live</button>
      <input type="range" id="scrub" min="0" max="1000" value="1000">
      <span class="clock" id="clock">0:00</span>
      <label class="status"><input type="checkbox" id="trails" checked> trails</label>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>People <span id="peopleCount" style="text-transform:none;letter-spacing:0;font-weight:400"></span></h2>
      <div id="people"><div class="empty">no one identified yet</div></div>
    </div>
    <div class="panel">
      <h2>Speech</h2>
      <ul id="speech"></ul>
      <audio id="player"></audio>
    </div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let D = null, live = true, playing = false, cursor = 0, lastFetch = 0;
// Wall-clock alignment so Live mode keeps advancing between session polls.
let serverNow = 0, clientAtFetch = 0;
const TRAIL_S = 25;

const PALETTE = ["#58a6ff","#7ee787","#ffa657","#d2a8ff","#79c0ff",
                 "#ff7b72","#f0883e","#a5d6ff","#56d364"];
const colorCache = {};
function colorFor(pid){
  if(!(pid in colorCache)){
    colorCache[pid] = PALETTE[Object.keys(colorCache).length % PALETTE.length];
  }
  return colorCache[pid];
}
const fmt = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`;

async function poll(){
  try{
    const r = await fetch("/api/session");
    D = await r.json();
    lastFetch = Date.now();
    serverNow = D.now;
    clientAtFetch = Date.now() / 1000;
    $("#status").innerHTML = D.running
      ? '<span class="live">● recording</span>'
      : "session ended";
    const nPeople = (D.people && D.people.length) || Object.keys(D.labels).length;
    $("#units").textContent =
      `${D.speech.length} utterances · ${nPeople} people`;
    const cov = Math.round((D.background_coverage || 0) * 100);
    $("#maptype").innerHTML = mode3d
      ? (D.running && live
          ? '<span class="live">— live camera</span>'
          : `<span style="color:var(--dim)">— room ${cov}% recovered</span>`)
      : (D.calibrated
          ? '<span style="color:var(--dim)">— true plan view, metres</span>'
          : '<span class="warn">— approximate: floor not calibrated</span>');
    if(live) cursor = liveCursor();
    refreshPlate();
    draw();
  }catch(e){ $("#status").textContent = "disconnected"; }
}
const duration = () => D ? Math.max(1, (D.ended_at || estimatedNow()) - D.started_at) : 1;
const t0 = () => D ? D.started_at : 0;
// Prefer wall-clock advance while recording so Live does not freeze between polls.
const estimatedNow = () => {
  if(!D) return 0;
  if(D.ended_at) return D.ended_at;
  if(serverNow) return serverNow + (Date.now()/1000 - clientAtFetch);
  return D.now;
};
const liveCursor = () => Math.max(0, estimatedNow() - t0());

let mode3d = true;
const plate = new Image();
let plateReady = false, plateStamp = 0;
plate.onload = () => { plateReady = true; draw(); };

// Live camera preview. Separate from the empty-room plate: plate is for
// scrubbing reconstructions; this is the actual video feed while recording.
const liveFrame = new Image();
let liveFrameReady = false, liveFrameStamp = 0, liveFrameInflight = false;
liveFrame.onload = () => { liveFrameReady = true; if(live && mode3d) draw(); };

function refreshPlate(){
  // The plate keeps improving as more of the room is seen without people in it,
  // so re-fetch it occasionally rather than caching the first version forever.
  if(!D || !D.background) return;
  if(Date.now() - plateStamp < 8000) return;
  plateStamp = Date.now();
  plate.src = D.background + "?t=" + plateStamp;
}

function refreshLiveFrame(){
  if(!D || !D.running || !live || !mode3d) return;
  if(liveFrameInflight) return;
  if(Date.now() - liveFrameStamp < 80) return;  // ~12 fps cap
  liveFrameInflight = true;
  liveFrameStamp = Date.now();
  const img = new Image();
  img.onload = () => {
    liveFrame.src = img.src;
    liveFrameReady = true;
    liveFrameInflight = false;
    if(live && mode3d) draw();
  };
  img.onerror = () => { liveFrameInflight = false; };
  img.src = "/api/frame.jpg?t=" + liveFrameStamp;
}

function draw(){
  if(mode3d) drawView(); else drawMap();
  drawTimeline(); drawList(); drawPeople(); syncControls();
}

function drawPeople(){
  const box = $("#people");
  if(!D){ box.innerHTML = '<div class="empty">connecting…</div>'; return; }
  const people = D.people || [];
  $("#peopleCount").textContent = people.length
    ? `— ${people.length} this session` : "";
  if(!people.length){
    box.innerHTML = '<div class="empty">no one identified yet — face the camera</div>';
    return;
  }
  box.innerHTML = people.map(p => {
    const col = colorFor(p.pid);
    const here = !!p.present;
    const initial = (p.label || p.pid || "?").trim().slice(0,1).toUpperCase();
    const face = p.photo
      ? `<img class="face" src="${p.photo}?t=${Math.floor(p.last_seen||0)}" alt=""
              style="border-color:${col}"
              onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'face placeholder',textContent:'${initial}',style:'border-color:${col}'}))">`
      : `<div class="face placeholder" style="border-color:${col}">${initial}</div>`;
    const looking = !!p.looking;
    const lookScore = (p.looking_score != null) ? Number(p.looking_score).toFixed(2) : null;
    const badges = [
      `<span class="badge">${here ? "in room" : "away"}</span>`,
      looking
        ? `<span class="badge looking">looking${lookScore ? " · "+lookScore : ""}</span>`
        : (here ? `<span class="badge">not looking</span>` : ""),
    ].filter(Boolean).join("");
    return `<div class="person ${here?"here":""}${looking?" looking":""}" data-pid="${p.pid}">
      ${face}
      <div class="meta">
        <div class="name" style="color:${col}">${p.label || p.pid}</div>
        <div class="id">${p.pid} · ${p.sightings||0} looks</div>
      </div>
      <div class="badges">${badges}</div>
    </div>`;
  }).join("");
}

// COCO-17 skeleton. Drawing the links rather than loose dots is what makes a
// figure read as a person rather than a constellation.
const LINKS = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],
               [11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const KP_MIN = 0.4;

function drawView(){
  const c = $("#view"), g = c.getContext("2d");
  const w = c.width, h = c.height;
  g.clearRect(0,0,w,h);
  if(!D) return;

  const fs = D.frame_size || [1280,720];
  // Letterbox so the image is never stretched.
  const s = Math.min(w/fs[0], h/fs[1]);
  const ox = (w - fs[0]*s)/2, oy = (h - fs[1]*s)/2;
  const T = (x,y) => [ox + x*s, oy + y*s];

  // Live recording + Live button: show the real camera feed. Scrubbing and
  // review still use the empty-room plate + pose reconstruction.
  const showLive = !!(D.running && live && liveFrameReady);
  if(showLive){
    g.drawImage(liveFrame, ox, oy, fs[0]*s, fs[1]*s);
  } else if(plateReady){
    g.drawImage(plate, ox, oy, fs[0]*s, fs[1]*s);
    // Dim it so the figures read clearly on top.
    g.fillStyle = "rgba(11,14,19,.34)"; g.fillRect(ox, oy, fs[0]*s, fs[1]*s);
  } else {
    g.fillStyle = "#12161d"; g.fillRect(ox, oy, fs[0]*s, fs[1]*s);
    g.fillStyle = "#5b6572"; g.font = "13px -apple-system,sans-serif";
    g.textAlign = "center";
    g.fillText(D.running && live ? "waiting for live camera…"
      : (D.background ? "loading the room…"
        : "building the empty-room plate — keep the camera still"), w/2, h/2);
    g.textAlign = "left";
  }

  const now = t0() + cursor;
  // Nearest pose per person at the scrub position; nothing if they were not
  // there. A stale skeleton left on screen would be a person who has left.
  const latest = {};
  for(const p of D.poses){
    if(p.t > now || now - p.t > 1.2) continue;
    const cur = latest[p.pid];
    if(!cur || p.t > cur.t) latest[p.pid] = p;
  }

  // On the live feed the pixels already show the person - only draw labels /
  // boxes / voting tags so we do not paint a giant blue head on top of them.
  // Reconstruction mode still draws the full stick figure on the empty plate.
  for(const pid in latest){
    const p = latest[pid], col = colorFor(pid);
    const speaking = D.speech.some(sp => sp.pid === pid && sp.start <= now && sp.end >= now);

    if(!showLive && p.kp){
      const K = p.kp;
      const pt = i => (K[i] && K[i][2] >= KP_MIN) ? T(K[i][0], K[i][1]) : null;

      // Torso as a filled shape, so the figure has a body and not just sticks.
      const ls=pt(5), rs=pt(6), lh=pt(11), rh=pt(12);
      if(ls&&rs&&lh&&rh){
        g.beginPath(); g.moveTo(...ls); g.lineTo(...rs); g.lineTo(...rh); g.lineTo(...lh);
        g.closePath(); g.fillStyle = col + "44"; g.fill();
      }
      g.strokeStyle = col; g.lineWidth = 3.5; g.lineCap = "round"; g.lineJoin = "round";
      for(const [a,bq] of LINKS){
        const A = pt(a), B = pt(bq);
        if(A && B){ g.beginPath(); g.moveTo(...A); g.lineTo(...B); g.stroke(); }
      }
      // Head sized from the shoulders, so it shrinks correctly with distance.
      const nose = pt(0);
      if(nose){
        let r = 11;
        if(ls&&rs) r = Math.max(7, Math.hypot(ls[0]-rs[0], ls[1]-rs[1]) * 0.34);
        g.beginPath(); g.arc(nose[0], nose[1]-r*0.25, r, 0, 7);
        g.fillStyle = col; g.fill();
      }
      for(const i of [9,10]){          // hands, so a raised one is obvious
        const H = pt(i);
        if(H){ g.beginPath(); g.arc(H[0], H[1], 4.5, 0, 7); g.fillStyle = col; g.fill(); }
      }
    } else if(p.box){
      // Live feed: outline the person. Reconstruction without keypoints: same.
      const [x1,y1,x2,y2] = p.box, A = T(x1,y1), B = T(x2,y2);
      g.strokeStyle = col; g.lineWidth = 2;
      g.strokeRect(A[0], A[1], B[0]-A[0], B[1]-A[1]);
    }

    const box = p.box || [0,0,0,0];
    const [lx, ly] = T((box[0]+box[2])/2, box[1]);
    if(p.up){
      g.fillStyle = "#ffb84d"; g.font = "bold 12px -apple-system,sans-serif";
      g.textAlign = "center"; g.fillText("VOTING", lx, ly - 22); g.textAlign = "left";
    }
    if(p.looking){
      g.fillStyle = "#58a6ff"; g.font = "bold 11px -apple-system,sans-serif";
      g.textAlign = "center";
      g.fillText("LOOKING", lx, ly - (p.up ? 38 : 22));
      g.textAlign = "left";
    }
    if(speaking){
      g.beginPath(); g.arc(lx, ly - 8, 6, 0, 7); g.fillStyle = "#7ee787"; g.fill();
    }
    g.fillStyle = "#e6edf3"; g.font = "12px -apple-system,sans-serif";
    g.textAlign = "center";
    g.fillText(D.labels[pid] || pid, lx, ly - 34 - (p.looking || p.up ? 14 : 0));
    g.textAlign = "left";
  }

  if(!Object.keys(latest).length && (showLive || plateReady)){
    g.fillStyle = "#5b6572"; g.font = "12px -apple-system,sans-serif";
    g.textAlign = "center"; g.fillText("nobody in the room at this moment", w/2, h - 14);
    g.textAlign = "left";
  }

  // Bottom status bar — same info as the OpenCV overlay.
  drawHud(g, ox, oy, fs[0]*s, fs[1]*s);
}

function drawHud(g, ox, oy, vw, vh){
  if(!D) return;
  const hud = D.hud || {};
  const strip = Math.max(44, Math.round(vh * 0.09));
  const top = oy + vh - strip;
  g.fillStyle = "rgba(11,14,19,0.72)";
  g.fillRect(ox, top, vw, strip);

  const fps = (hud.fps != null ? hud.fps : 0).toFixed(1);
  const people = hud.people != null ? hud.people : Object.keys(latestPeople()).length;
  const named = hud.named != null ? hud.named : (D.people || []).filter(p => p.present).length;
  const looking = hud.looking != null ? hud.looking
    : (D.people || []).filter(p => p.looking).length;
  const gallery = hud.gallery != null ? hud.gallery : (D.people || []).length;
  const floor = hud.floor || (D.calibrated ? "metres" : "uncalibrated");
  const line1 = `${fps} fps   people ${people} (${named} named)   looking ${looking}   gallery ${gallery}   floor: ${floor}`;
  const line2 = "[1-9] name person    [s] save    [q] quit";

  g.textAlign = "left";
  g.fillStyle = "#ebebeb";
  g.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  g.fillText(line1, ox + 12, top + strip * 0.42);

  const voting = hud.voting || [];
  if(voting.length){
    const tally = `VOTING (${voting.length}): ${voting.join(", ")}`;
    g.fillStyle = "#ffb84d";
    g.font = "bold 12px -apple-system,sans-serif";
    g.textAlign = "right";
    g.fillText(tally, ox + vw - 12, top + strip * 0.42);
    g.textAlign = "left";
  }

  g.fillStyle = "#969696";
  g.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  g.fillText(line2, ox + 12, top + strip * 0.78);
}

function latestPeople(){
  if(!D) return {};
  const now = t0() + cursor;
  const latest = {};
  for(const p of (D.poses || [])){
    if(p.t > now || now - p.t > 1.2) continue;
    if(!latest[p.pid] || p.t > latest[p.pid].t) latest[p.pid] = p;
  }
  return latest;
}

function syncControls(){
  if(live) cursor = liveCursor();
  $("#scrub").value = (cursor / duration()) * 1000;
  $("#clock").textContent = `${fmt(cursor)} / ${fmt(duration())}`;
  $("#liveBtn").classList.toggle("on", live);
  $("#play").textContent = playing ? "❚❚ Pause" : "▶ Play";
}

function project(b, w, h){
  // Preserve aspect so a square room looks square, whatever the canvas is.
  const pad = 34;
  const sx = (w - pad*2) / Math.max(1e-6, b.xmax - b.xmin);
  const sy = (h - pad*2) / Math.max(1e-6, b.ymax - b.ymin);
  const s = Math.min(sx, sy);
  const ox = (w - (b.xmax - b.xmin) * s) / 2;
  const oy = (h - (b.ymax - b.ymin) * s) / 2;
  // y is flipped for metres (further away = up the screen); in raw pixel mode
  // the value already grows downward, so it is left alone.
  return (x, y) => [ox + (x - b.xmin) * s,
                    D.calibrated ? h - oy - (y - b.ymin) * s : oy + (y - b.ymin) * s];
}

function drawMap(){
  const c = $("#map"), g = c.getContext("2d");
  const w = c.width, h = c.height;
  g.clearRect(0,0,w,h);
  if(!D) return;
  const b = D.bounds, P = project(b, w, h);

  // grid
  g.strokeStyle = "#1b222c"; g.lineWidth = 1; g.font = "10px ui-monospace,monospace";
  g.fillStyle = "#3d4650";
  if(D.calibrated){
    for(let x = Math.ceil(b.xmin); x <= b.xmax; x++){
      const [px] = P(x, b.ymin); g.beginPath();
      g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
      g.fillText(x + "m", px + 3, h - 6);
    }
    for(let y = Math.ceil(b.ymin); y <= b.ymax; y++){
      const [,py] = P(b.xmin, y); g.beginPath();
      g.moveTo(0, py); g.lineTo(w, py); g.stroke();
      g.fillText(y + "m", 4, py - 3);
    }
  }

  // What the camera can see, drawn under everything else. This comes straight
  // from the homography and does not depend on the focal-length estimate, so
  // unlike the camera marker it is as accurate as the calibration itself.
  if(D.fov && D.fov.length > 2){
    g.beginPath();
    D.fov.forEach(([x,y], i) => { const [px,py] = P(x,y); i ? g.lineTo(px,py) : g.moveTo(px,py); });
    g.closePath();
    g.fillStyle = "rgba(88,166,255,.07)"; g.fill();
    g.strokeStyle = "rgba(88,166,255,.28)"; g.lineWidth = 1; g.stroke();
  }

  if(D.camera){
    const [cxp, cyp] = P(D.camera.ground[0], D.camera.ground[1]);
    g.beginPath(); g.arc(cxp, cyp, 8, 0, 7);
    g.fillStyle = "#e6edf3"; g.fill();
    g.strokeStyle = "#0b0e13"; g.lineWidth = 2; g.stroke();
    // A dashed ring when the position was only estimated, so a marker that is
    // 25cm out does not read as surveyed.
    if(D.camera.source !== "measured"){
      g.beginPath(); g.arc(cxp, cyp, 13, 0, 7);
      g.strokeStyle = "rgba(230,237,243,.45)"; g.lineWidth = 1;
      g.setLineDash([3,3]); g.stroke(); g.setLineDash([]);
    }
    g.fillStyle = "#8b949e"; g.font = "11px -apple-system,sans-serif";
    const note = D.camera.source === "measured" ? "camera"
               : `camera (${D.camera.source}, ±25cm)`;
    g.fillText(note, cxp + 14, cyp + 4);
    g.fillText(`${D.camera.height.toFixed(2)}m high`, cxp + 14, cyp + 17);
  }

  const now = t0() + cursor;
  const byPid = {};
  for(const [t, pid, x, y] of D.positions) (byPid[pid] ||= []).push([t, x, y]);

  const showTrails = $("#trails").checked;
  for(const pid in byPid){
    const pts = byPid[pid].filter(p => p[0] <= now);
    if(!pts.length) continue;
    const col = colorFor(pid);

    if(showTrails){
      g.lineWidth = 2; g.lineJoin = "round";
      const trail = pts.filter(p => p[0] >= now - TRAIL_S);
      // Fade the trail so the direction of travel is readable.
      for(let i = 1; i < trail.length; i++){
        const a = (i / trail.length) * 0.85 + 0.08;
        g.globalAlpha = a; g.strokeStyle = col;
        g.beginPath();
        g.moveTo(...P(trail[i-1][1], trail[i-1][2]));
        g.lineTo(...P(trail[i][1], trail[i][2]));
        g.stroke();
      }
      g.globalAlpha = 1;
    }

    // Only draw the person if they were actually seen recently.
    const last = pts[pts.length-1];
    if(now - last[0] > 2.0) continue;
    const [px, py] = P(last[1], last[2]);

    const speaking = D.speech.some(s => s.pid === pid && s.start <= now && s.end >= now);
    const voting = D.gestures.some(gg => gg.pid === pid && gg.start <= now &&
                                         (gg.end === null || gg.end >= now));
    if(speaking){
      g.beginPath(); g.arc(px, py, 17, 0, 7); g.fillStyle = "rgba(126,231,135,.18)"; g.fill();
      g.strokeStyle = "#7ee787"; g.lineWidth = 2; g.stroke();
    }
    if(voting){
      g.beginPath(); g.arc(px, py, 23, 0, 7);
      g.strokeStyle = "#ffb84d"; g.lineWidth = 2; g.setLineDash([3,3]); g.stroke();
      g.setLineDash([]);
    }
    g.beginPath(); g.arc(px, py, 7, 0, 7); g.fillStyle = col; g.fill();
    g.fillStyle = "#e6edf3"; g.font = "12px -apple-system,sans-serif";
    let name = D.labels[pid] || pid;
    if(D.camera && D.calibrated){
      const dx = last[1] - D.camera.ground[0], dy = last[2] - D.camera.ground[1];
      name += `  ${Math.hypot(dx,dy).toFixed(1)}m`;
    }
    g.fillText(name, px + 12, py + 4);
  }

  // legend
  $("#legend").innerHTML = Object.keys(byPid).map(pid =>
    `<span><i class="swatch" style="background:${colorFor(pid)}"></i>${D.labels[pid]||pid}</span>`
  ).join("") + '<span><i class="swatch" style="background:#7ee787"></i>speaking</span>' +
    '<span><i class="swatch" style="background:#ffb84d"></i>hand up</span>' +
    (D.camera ? '<span><i class="swatch" style="background:#e6edf3"></i>camera</span>' : '') +
    (D.fov ? '<span><i class="swatch" style="background:rgba(88,166,255,.4)"></i>camera view</span>' : '');
}

function drawTimeline(){
  const c = $("#timeline"), g = c.getContext("2d");
  const w = c.width, h = c.height;
  g.clearRect(0,0,w,h);
  if(!D) return;
  const dur = duration(), X = t => (t / dur) * w;

  const pids = Object.keys(D.labels);
  const rowH = Math.max(12, Math.min(22, (h - 26) / Math.max(1, pids.length)));

  g.font = "10px -apple-system,sans-serif";
  pids.forEach((pid, i) => {
    const y = 4 + i * rowH;
    g.fillStyle = "#141a22"; g.fillRect(0, y, w, rowH - 3);
    g.fillStyle = "#4c565f";
    g.fillText(D.labels[pid] || pid, 4, y + rowH - 8);

    for(const s of D.speech){
      if(s.pid !== pid) continue;
      const x1 = X(s.start - t0()), x2 = X(s.end - t0());
      g.fillStyle = "#7ee787";
      g.fillRect(x1, y + 1, Math.max(2, x2 - x1), rowH - 5);
    }
    for(const gg of D.gestures){
      if(gg.pid !== pid) continue;
      const x1 = X(gg.start - t0());
      const x2 = X((gg.end || (t0() + dur)) - t0());
      g.fillStyle = "rgba(255,184,77,.75)";
      g.fillRect(x1, y + rowH - 5, Math.max(2, x2 - x1), 3);
    }
  });

  // Unattributed speech gets its own lane rather than being hidden.
  const orphan = D.speech.filter(s => !s.pid);
  if(orphan.length){
    const y = h - 20;
    g.fillStyle = "#4c565f"; g.fillText("unattributed", 4, y + 11);
    for(const s of orphan){
      g.fillStyle = "rgba(139,148,158,.7)";
      g.fillRect(X(s.start - t0()), y + 1, Math.max(2, X(s.end - s.start)), 9);
    }
  }

  const cx = X(cursor);
  g.strokeStyle = "#58a6ff"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, h); g.stroke();
}

function drawList(){
  const ul = $("#speech");
  if(!D || !D.speech.length){
    ul.innerHTML = '<div class="empty">nothing heard yet — say something</div>';
    return;
  }
  const now = t0() + cursor;
  ul.innerHTML = D.speech.slice().reverse().map(s => {
    const active = s.start <= now && s.end >= now;
    const col = s.pid ? colorFor(s.pid) : "#8b949e";
    return `<li class="${active?"active":""}" data-id="${s.id}" data-start="${s.start}">
      <i class="swatch" style="background:${col}"></i>
      <span class="t">${fmt(s.start - t0())}</span>
      <span class="who">${s.label}</span>
      <span class="status">${(s.end-s.start).toFixed(1)}s</span>
      <span class="basis">${s.basis}</span></li>`;
  }).join("");
  ul.querySelectorAll("li").forEach(li => li.onclick = () => {
    const s = D.speech[+li.dataset.id];
    live = false; playing = false;
    cursor = s.start - t0();
    if(s.audio){ const p = $("#player"); p.src = s.audio; p.play().catch(()=>{}); }
    draw();
  });
}

$("#scrub").oninput = e => {
  live = false; cursor = (e.target.value/1000) * duration(); draw();
};
$("#liveBtn").onclick = () => {
  live = !live;
  if(live){ playing = false; cursor = liveCursor(); refreshLiveFrame(); }
  draw();
};
$("#play").onclick = () => { playing = !playing; if(playing) live = false; draw(); };
$("#trails").onchange = draw;
$("#tabView").onclick = () => { mode3d = true; $("#view").hidden = false;
  $("#map").hidden = true; $("#tabView").classList.add("on");
  $("#tabMap").classList.remove("on"); draw(); };
$("#tabMap").onclick = () => { mode3d = false; $("#view").hidden = true;
  $("#map").hidden = false; $("#tabMap").classList.add("on");
  $("#tabView").classList.remove("on"); draw(); };
$("#timeline").onclick = e => {
  const r = e.target.getBoundingClientRect();
  live = false;
  cursor = ((e.clientX - r.left) / r.width) * duration();
  draw();
};

setInterval(() => {
  if(playing){
    cursor += 0.1;
    if(cursor >= duration()){ cursor = duration(); playing = false; }
    draw();
  } else if(live && D && D.running){
    // Keep the clock and labels advancing between session polls.
    cursor = liveCursor();
    syncControls();
  }
}, 100);
// Live camera frames ~12 fps; session metadata (speech, poses) a bit slower.
setInterval(refreshLiveFrame, 90);
setInterval(poll, 500);
poll();
refreshLiveFrame();
</script></body></html>
"""
