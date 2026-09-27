// Client : micro -> WebSocket (PCM16 16 kHz), WebSocket -> lecteur (PCM16 24 kHz),
// et la copie : ce que tu dis à l'encre bleue, les corrections du prof au stylo rouge.

const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  mode: "handsfree",
  teacher: "",
  userId: null,
  subject: null,
  users: [],
  subjects: [],
  player: null,
  playing: false,
  speakingTurn: null,
  flushed: new Set(),
  doneTurns: new Set(),
  pendingUser: null,
  userLines: new Map(),
  teacherLines: new Map(),
  pttDown: false,
  started: false,
  fixCount: 0,
};

// --- affichage --------------------------------------------------------------------
function setStatus(text, micState = "") {
  $("status").textContent = text;
  $("mic").dataset.state = micState;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function addLine(kind, who, text) {
  const li = document.createElement("li");
  li.className = kind;
  li.innerHTML = `<span class="who"></span><span class="said"></span>`;
  li.querySelector(".who").textContent = who;
  li.querySelector(".said").textContent = text;
  $("conversation").appendChild(li);
  li.scrollIntoView({ block: "end" });
  return li;
}

function correctedHtml(text, fixes) {
  const lower = text.toLowerCase();
  const ranges = [];
  for (const fix of fixes) {
    const needle = fix.original.toLowerCase();
    let from = 0;
    while (true) {
      const start = lower.indexOf(needle, from);
      if (start < 0) break;
      const end = start + needle.length;
      if (!ranges.some((r) => start < r.end && end > r.start)) {
        ranges.push({ start, end, fix });
        break;
      }
      from = start + 1;
    }
  }
  ranges.sort((a, b) => a.start - b.start);
  let html = "";
  let pos = 0;
  for (const { start, end, fix } of ranges) {
    html += escapeHtml(text.slice(pos, start));
    html += `<span class="err" title="${escapeHtml(fix.explain_fr || "")}"><s>${escapeHtml(text.slice(start, end))}</s>` +
      `<span class="fix">${escapeHtml(fix.corrected)}</span></span>`;
    pos = end;
  }
  return html + escapeHtml(text.slice(pos));
}

function addFixesToList(fixes) {
  const list = $("fix-list");
  if (!state.fixCount) list.innerHTML = "";
  for (const fix of fixes) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="fix-pair"><s>${escapeHtml(fix.original)}</s><span class="fix-to">${escapeHtml(fix.corrected)}</span></span>` +
      `<span class="fix-explain">${escapeHtml(fix.explain_fr || "")}</span>`;
    list.prepend(li);
    state.fixCount++;
  }
}

async function loadRecurring() {
  try {
    const stats = await getJson(`/api/stats?user=${state.userId}&subject=${encodeURIComponent(state.subject)}`);
    const list = $("recurring");
    if (!stats.top_errors.length) {
      list.innerHTML = '<li class="empty">Rien à revoir pour l’instant.</li>';
      return;
    }
    list.innerHTML = "";
    for (const e of stats.top_errors.slice(0, 8)) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="fix-pair"><s>${escapeHtml(e.example_original || "")}</s>` +
        `<span class="fix-to">${escapeHtml(e.example_corrected || "")}</span></span> ` +
        `<span class="count">${e.count} fois</span><span class="fix-explain">${escapeHtml(e.explain_fr || "")}</span>`;
      list.appendChild(li);
    }
  } catch {
    // le panneau reste sur son message par défaut
  }
}

function showContext(ev) {
  const pct = Math.min(100, Math.round((100 * ev.used) / ev.max));
  const bar = $("ctx-bar");
  $("ctx-fill").style.width = `${pct}%`;
  bar.setAttribute("aria-valuenow", pct);
  bar.classList.toggle("full", pct >= 85);
  const n = (v) => v.toLocaleString("fr-FR");
  $("ctx-text").textContent = `${n(ev.used)} / ${n(ev.max)} tokens · ${pct} %`;
  const note = $("ctx-note");
  note.hidden = !ev.trims;
  note.textContent = ev.trims === 1
    ? "Conversation coupée une fois : le prof a oublié la plus ancienne moitié."
    : `Conversation coupée ${ev.trims} fois : le prof n’en garde que la partie récente.`;
}

function showLatency(lat) {
  const ms = (v) => (v === undefined ? "–" : `${v} ms`);
  $("lat-audio").textContent = ms(lat.first_audio);
  $("lat-stt").textContent = ms(lat.stt);
  $("lat-llm").textContent = ms(lat.llm_first_token);
}

// --- WebSocket ----------------------------------------------------------------------
function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function connect() {
  const previous = state.ws;
  state.ws = null;
  previous?.close();
  $("start").disabled = true;
  if (!state.userId || !state.subject) return;
  const query = `user=${state.userId}&subject=${encodeURIComponent(state.subject)}`;
  const ws = new WebSocket(`ws://${location.host}/ws?${query}`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    if (!state.started) {
      setStatus("Prêt à commencer");
      $("start").disabled = false;
      return;
    }
    // reconnexion après un redémarrage du serveur : nouvelle séance, même page
    state.flushed.clear();
    state.doneTurns.clear();
    state.userLines.clear();   // les numéros de tour repartent de 1
    state.teacherLines.clear();
    state.pendingUser = null;
    state.player?.port.postMessage({ type: "flush" });
    state.playing = false;
    $("mic").disabled = false;
    send({ type: "mode", value: state.mode });
    send({ type: "start" });
  };
  ws.onclose = (ev) => {
    if (ws !== state.ws) return; // remplacée : changement de profil ou de matière
    $("mic").disabled = true;
    $("start").disabled = true;
    if (state.ended) return;
    if (ev.code === 4404) {
      setStatus("Profil ou matière introuvable : recharge la page.");
      return;
    }
    setStatus("Connexion au serveur perdue, reconnexion…");
    setTimeout(connect, 2000);
  };
  ws.onmessage = ({ data }) => (typeof data === "string" ? onEvent(JSON.parse(data)) : onAudio(data));
}

// --- profil et matière -------------------------------------------------------------------
const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* stockage indisponible : on redemandera */ }
  },
};

async function getJson(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function jsonBody(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

async function loadPicker() {
  try {
    [state.users, state.subjects] = await Promise.all([getJson("/api/users"), getJson("/api/subjects")]);
  } catch {
    setStatus("Serveur injoignable : lance run.bat puis recharge la page.");
    return;
  }
  const savedUser = Number(store.get("profs.user"));
  const user = state.users.find((u) => u.id === savedUser) || (state.users.length === 1 ? state.users[0] : null);
  const savedSubject = store.get("profs.subject");
  const subject = state.subjects.find((s) => s.id === savedSubject)
    || (state.subjects.length === 1 ? state.subjects[0] : null);
  state.userId = user?.id ?? null;
  if (subject) useSubject(subject);
  renderProfiles();
  renderSubjects();
  connect();
  if (!state.users.length) $("new-name").focus();
}

function renderProfiles() {
  const list = $("profiles");
  list.innerHTML = "";
  for (const user of state.users) {
    const li = document.createElement("li");
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "choice";
    pick.textContent = user.name;
    pick.setAttribute("aria-pressed", String(user.id === state.userId));
    pick.onclick = () => selectUser(user.id);
    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "rename";
    rename.textContent = "Renommer";
    rename.setAttribute("aria-label", `Renommer ${user.name}`);
    rename.onclick = () => startRename(li, user);
    li.append(pick, rename);
    list.appendChild(li);
  }
}

function startRename(li, user) {
  const input = document.createElement("input");
  input.value = user.name;
  input.maxLength = 40;
  input.setAttribute("aria-label", `Nouveau nom pour ${user.name}`);
  li.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (save && name && name !== user.name) {
      try {
        Object.assign(user, await getJson(`/api/users/${user.id}`, jsonBody("PATCH", { name })));
        if (user.id === state.userId) connect(); // le prof doit connaître le nouveau prénom
      } catch {
        setStatus("Renommage refusé : 40 caractères au plus.");
      }
    }
    renderProfiles();
  };
  input.onkeydown = (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  };
  input.onblur = () => finish(true);
}

function renderSubjects() {
  const list = $("subjects");
  list.innerHTML = "";
  for (const subject of state.subjects) {
    const li = document.createElement("li");
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "choice subject";
    pick.setAttribute("aria-pressed", String(subject.id === state.subject));
    pick.innerHTML = `<span class="subject-title">${escapeHtml(subject.title)} avec ${escapeHtml(subject.teacher)}</span>`
      + `<span class="subject-desc">${escapeHtml(subject.description)}</span>`;
    pick.onclick = () => selectSubject(subject.id);
    li.appendChild(pick);
    list.appendChild(li);
  }
}

function useSubject(subject) {
  state.subject = subject.id;
  state.teacher = subject.teacher;
  $("teacher-name").textContent = subject.teacher;
  setMode(state.mode);
}

function selectUser(id) {
  state.userId = id;
  store.set("profs.user", String(id));
  renderProfiles();
  connect();
}

function selectSubject(id) {
  useSubject(state.subjects.find((s) => s.id === id));
  store.set("profs.subject", id);
  renderSubjects();
  connect();
}

async function createProfile(e) {
  e.preventDefault();
  const name = $("new-name").value.trim();
  if (!name) return;
  try {
    const user = await getJson("/api/users", jsonBody("POST", { name }));
    state.users.unshift(user);
    $("new-name").value = "";
    selectUser(user.id);
  } catch {
    setStatus("Création refusée : 40 caractères au plus.");
  }
}

function onAudio(buf) {
  const view = new DataView(buf);
  const turn = view.getUint32(0, true);
  const seg = view.getUint32(4, true);
  if (state.flushed.has(turn) || !state.player) return;
  const pcm = new Int16Array(buf, 8);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
  state.player.port.postMessage({ type: "push", turn, seg, samples }, [samples.buffer]);
  state.playing = true;
  state.speakingTurn = turn;
  setStatus(`${state.teacher} parle`, "speaking");
}

function listeningStatus() {
  if (state.pttDown) setStatus("Je t'écoute (touche maintenue)", "held");
  else setStatus(state.mode === "handsfree" ? "Je t'écoute" : "Maintiens Espace pour parler", "listening");
}

function onEvent(ev) {
  switch (ev.type) {
    case "hello":
      state.teacher = ev.teacher;
      $("teacher-name").textContent = ev.teacher;
      loadRecurring();
      break;
    case "listening":
      if (!state.pendingUser) state.pendingUser = addLine("user pending", "Toi", "…");
      listeningStatus();
      break;
    case "partial":
      if (!state.pendingUser) state.pendingUser = addLine("user pending", "Toi", "");
      state.pendingUser.querySelector(".said").textContent = ev.text;
      break;
    case "user_final": {
      const li = state.pendingUser || addLine("user", "Toi", "");
      li.classList.remove("pending");
      li.querySelector(".said").textContent = ev.text;
      state.userLines.set(ev.turn, li);
      state.pendingUser = null;
      break;
    }
    case "noinput":
      state.pendingUser?.remove();
      state.pendingUser = null;
      listeningStatus();
      break;
    case "assistant_start":
      state.teacherLines.set(ev.turn, addLine("teacher-line", state.teacher, ""));
      setStatus(`${state.teacher} réfléchit`, "thinking");
      break;
    case "assistant_delta": {
      const li = state.teacherLines.get(ev.turn);
      if (li) {
        li.querySelector(".said").textContent += ev.text;
        li.scrollIntoView({ block: "end" });
      }
      break;
    }
    case "fixes": {
      const li = state.userLines.get(ev.turn);
      if (li && ev.fixes.length) li.querySelector(".said").innerHTML = correctedHtml(ev.user_text, ev.fixes);
      if (ev.fixes.length) addFixesToList(ev.fixes);
      break;
    }
    case "assistant_done":
      state.doneTurns.add(ev.turn);
      if (!state.playing) {
        send({ type: "played", turn: ev.turn });
        listeningStatus();
      }
      break;
    case "flush":
      state.flushed.add(ev.turn);
      state.player?.port.postMessage({ type: "flush" });
      state.playing = false;
      state.teacherLines.get(ev.turn)?.classList.add("interrupted");
      break;
    case "context":
      showContext(ev);
      break;
    case "metrics":
      showLatency(ev.latency_ms);
      break;
    case "error":
      setStatus(ev.message);
      break;
    case "session_ended":
      showSummary(ev.summary);
      break;
  }
}

function onPlayerEvent({ data }) {
  if (data.type === "seg_start") {
    send({ type: "seg_start", turn: data.turn, seg: data.seg });
  } else if (data.type === "drained") {
    state.playing = false;
    if (state.doneTurns.has(state.speakingTurn)) {
      send({ type: "played", turn: state.speakingTurn });
      listeningStatus();
    }
  }
}

// --- audio ----------------------------------------------------------------------------
async function startSession() {
  if (state.ws?.readyState !== WebSocket.OPEN) return;
  $("start").disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const micCtx = new AudioContext({ latencyHint: "interactive" });
    const playCtx = new AudioContext({ sampleRate: 24000, latencyHint: "interactive" });
    await micCtx.audioWorklet.addModule("mic-worklet.js");
    await playCtx.audioWorklet.addModule("player-worklet.js");

    const mic = new AudioWorkletNode(micCtx, "mic-processor");
    const mute = micCtx.createGain();
    mute.gain.value = 0;
    micCtx.createMediaStreamSource(stream).connect(mic).connect(mute).connect(micCtx.destination);
    mic.port.onmessage = ({ data }) => {
      if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(data.pcm);
      $("mic-level").style.height = `${Math.min(100, data.rms * 500)}%`;
    };

    state.player = new AudioWorkletNode(playCtx, "player-processor", { outputChannelCount: [1] });
    state.player.connect(playCtx.destination);
    state.player.port.onmessage = onPlayerEvent;
  } catch (err) {
    $("start").disabled = false;
    setStatus(`Micro indisponible : ${err.message}. Autorise le micro puis réessaie.`);
    return;
  }
  $("welcome").remove();
  $("mic").disabled = false;
  $("end-session").disabled = false;
  state.started = true;
  send({ type: "mode", value: state.mode });
  send({ type: "start" });
  setStatus(`${state.teacher} réfléchit`, "thinking");
}

function showSummary(summary) {
  state.ended = true;
  $("mic").disabled = true;
  $("end-session").disabled = true;
  const text = summary?.summary || "Séance enregistrée.";
  const li = addLine("teacher-line", "Bilan", text + (summary?.level ? ` Niveau estimé : ${summary.level}.` : ""));
  const btn = document.createElement("button");
  btn.className = "start";
  btn.textContent = "Nouvelle séance";
  btn.onclick = () => location.reload();
  li.querySelector(".said").append(document.createElement("br"), btn);
  setStatus("Séance terminée");
}

// --- commandes ------------------------------------------------------------------------
function pttDown() {
  if (!state.started || state.pttDown || state.ended) return;
  state.pttDown = true;
  if (state.playing) {
    state.player.port.postMessage({ type: "flush" });
    state.playing = false;
  }
  send({ type: "ptt", state: "down" });
  listeningStatus();
}

function pttUp() {
  if (!state.pttDown) return;
  state.pttDown = false;
  send({ type: "ptt", state: "up" });
  setStatus(`${state.teacher} réfléchit`, "thinking");
}

function setMode(mode) {
  state.mode = mode;
  for (const b of document.querySelectorAll(".mode button")) {
    b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  }
  const teacher = state.teacher || "ton prof";
  $("mic-hint").textContent = mode === "handsfree"
    ? `Maintiens Espace pour parler à tout moment. Échap coupe ${teacher}.`
    : `Maintiens Espace ou le bouton micro pendant que tu parles. Échap coupe ${teacher}.`;
  send({ type: "mode", value: mode });
  if (state.started && !state.playing) listeningStatus();
}

// Espace et Échap pilotent le micro, sauf quand on tape dans un champ (prénom)
const typing = (e) => e.target instanceof Element && e.target.closest("input, textarea");

document.addEventListener("keydown", (e) => {
  if (typing(e)) return;
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    pttDown();
  } else if (e.code === "Escape") {
    send({ type: "stop" });
  }
});
document.addEventListener("keyup", (e) => {
  if (typing(e)) return;
  if (e.code === "Space") {
    e.preventDefault();
    pttUp();
  }
});

$("mic").addEventListener("pointerdown", pttDown);
$("mic").addEventListener("pointerup", pttUp);
$("mic").addEventListener("pointerleave", pttUp);
$("start").addEventListener("click", startSession);
$("new-profile").addEventListener("submit", createProfile);
$("end-session").addEventListener("click", () => {
  send({ type: "end_session" });
  setStatus("Claire rédige le bilan de la séance");
});
for (const b of document.querySelectorAll(".mode button")) b.addEventListener("click", () => setMode(b.dataset.mode));

loadPicker();
