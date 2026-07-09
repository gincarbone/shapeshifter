# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Alfa1 — 3-column coding-agent UI.

Single embedded HTML/CSS/JS string, same "no build step" convention as
wrapper_server._DASHBOARD_HTML, but kept in its own module and split into
separate CSS/JS constants for readability.
"""
from __future__ import annotations

_ALFA1_CSS = """
:root {
  --bg:#0f1117; --panel:#1a1d27; --border:#2a2d3e;
  --accent:#7c6af7; --green:#22c55e; --yellow:#eab308;
  --red:#ef4444; --text:#e2e8f0; --muted:#64748b;
  --font:'JetBrains Mono',Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font);font-size:12px;height:100vh;overflow:hidden}

/* ---- dark scrollbars (Firefox + WebKit/Chromium/Edge) ---- */
*{scrollbar-width:thin;scrollbar-color:var(--border) var(--bg)}
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:var(--bg)}
*::-webkit-scrollbar-thumb{background:var(--border);border-radius:5px}
*::-webkit-scrollbar-thumb:hover{background:var(--accent)}
*::-webkit-scrollbar-corner{background:var(--bg)}

/* ---- layout / columns ----
   Widths use vw (viewport-relative) instead of fixed px so the layout
   scales proportionally across resolutions — a 400px column is generous on
   a small laptop screen and tiny on a 4K monitor. min/max-width in px stay
   as safety rails only, so columns never become unusably narrow (small
   screen) or absurdly wide (ultra-wide screen) regardless of the vw value.
   col3's flex-basis mirrors col2's vw width so the two start equal-sized
   at load on any resolution, not just coincidentally at one fixed size;
   flex-grow:1 still lets col3 fill any leftover viewport width beyond
   col1+col2+col3. Once the user drags a resizer, that column switches to
   an explicit px width (see setupResizer) and no longer scales with the
   viewport — expected, matching how a manual resize should behave. */
#layout{display:flex;height:100vh}
#col1{width:16vw;min-width:160px;max-width:420px;background:var(--panel);border-right:1px solid var(--border);
      display:flex;flex-direction:column}
#col2{width:26vw;min-width:220px;max-width:640px;border-right:1px solid var(--border);
      display:flex;flex-direction:column}
#col3{flex:1 1 26vw;min-width:280px;display:flex;flex-direction:column}
.resizer{width:5px;cursor:col-resize;background:var(--border);flex-shrink:0}
.resizer:hover,.resizer.active{background:var(--accent)}

/* ---- column 1: tree + stats ---- */
#tree-header{padding:10px 12px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
#tree-header h1{font-size:13px;color:var(--accent);margin:0;letter-spacing:.5px}
#pick-btn{background:var(--accent);color:#fff;border:none;border-radius:4px;padding:5px 8px;font-size:10px;cursor:pointer}
#workspace-bar{padding:5px 12px;font-size:9px;color:var(--muted);border-bottom:1px solid var(--border);
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
#tree{flex:1;overflow:auto;padding:6px}
.tree-entry{padding:4px 6px;border-radius:4px;cursor:pointer;white-space:nowrap;font-size:11px;color:var(--text);
            display:flex;align-items:center;gap:6px}
.tree-entry:hover{background:rgba(124,106,247,.12)}
.tree-entry.dir{color:var(--text);font-weight:600}
.tree-entry.active{background:rgba(124,106,247,.25)}
.tree-icon{font-size:12px;line-height:1;flex-shrink:0}
#stats-cards{display:flex;flex-direction:column;gap:8px;padding:10px;border-top:1px solid var(--border);flex-shrink:0}
.stat-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px}
.stat-card .label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.stat-card .value{font-size:18px;color:var(--green);font-weight:700;margin-top:2px}

/* ---- column 2: tabs + code ---- */
#tabs{display:flex;overflow-x:auto;background:var(--panel);border-bottom:1px solid var(--border);flex-shrink:0}
.tab{padding:7px 12px;font-size:11px;color:var(--muted);border-right:1px solid var(--border);cursor:pointer;white-space:nowrap}
.tab.active{color:var(--text);background:var(--bg)}
.tab .close{margin-left:6px;opacity:.5}
.tab .close:hover{opacity:1;color:var(--red)}
#code-view{flex:1;overflow:auto;background:var(--bg);position:relative}
#code-view pre{margin:0;padding:12px;font-family:var(--font);font-size:12px;line-height:1.5;white-space:pre;outline:none}
#code-empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)}
.tok-kw{color:#c792ea}

/* ---- cyberpunk "materializing" reveal for agent-written code (revealCode) ---- */
#code-view.revealing::before{
  content:'';position:absolute;left:0;right:0;top:0;height:2px;z-index:2;pointer-events:none;
  background:linear-gradient(90deg,transparent,#22d3ee,var(--accent),transparent);
  box-shadow:0 0 10px 2px #22d3ee,0 0 18px 4px rgba(124,106,247,.5);
  animation:alfa1-scan 1.2s linear infinite;
}
@keyframes alfa1-scan{0%{top:0}100%{top:100%}}
.code-line{
  opacity:0;transform:translateX(-6px);
  animation:alfa1-line-in .32s ease-out forwards;
}
.code-line.code-line-static{opacity:1;transform:none;animation:none;filter:none}
@keyframes alfa1-line-in{
  0%{opacity:0;transform:translateX(-6px);
     filter:drop-shadow(0 0 5px #22d3ee) drop-shadow(0 0 10px rgba(124,106,247,.8))}
  70%{opacity:1;filter:drop-shadow(0 0 5px #22d3ee) drop-shadow(0 0 10px rgba(124,106,247,.8))}
  100%{opacity:1;transform:translateX(0);filter:none}
}
.reveal-cursor{display:inline-block;width:7px;height:1em;margin-left:1px;vertical-align:text-bottom;
               background:#22d3ee;box-shadow:0 0 6px #22d3ee,0 0 10px #22d3ee;
               animation:alfa1-blink .8s steps(1) infinite}
@keyframes alfa1-blink{50%{opacity:0}}
.tok-str{color:#c3e88d}
.tok-com{color:#546e7a;font-style:italic}
.tok-num{color:#f78c6c}

/* ---- column 3: activity bar + chat ---- */
#col3-topbar{padding:8px 12px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;
             align-items:center;flex-shrink:0;gap:10px}
#col3-topbar .title{font-size:11px;color:var(--accent);letter-spacing:.7px;font-weight:700;text-transform:uppercase;margin-right:auto}
#cancel-btn{background:transparent;border:1px solid var(--red);color:var(--red);border-radius:4px;
            padding:3px 8px;font-size:9px;cursor:pointer}
#cancel-btn.hidden{display:none}
.topbar-icon-btn{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:4px;
                  padding:3px 7px;font-size:11px;cursor:pointer;line-height:1}
.topbar-icon-btn:hover{border-color:var(--accent);color:var(--accent)}
#clear-sessions-btn:hover{border-color:var(--red);color:var(--red)}
.activity-dot{width:8px;height:8px;border-radius:50%;background:var(--border);flex-shrink:0}
.activity-dot.working{background:var(--yellow);animation:alfa1-pulse 1s infinite}
.activity-dot.ok{background:var(--green)}
.activity-dot.error{background:var(--red)}
@keyframes alfa1-pulse{0%,100%{opacity:1}50%{opacity:.3}}

#chat-log{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
.msg{border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.5;max-width:100%}
.msg-user{background:rgba(124,106,247,.12);align-self:flex-end}
.msg-assistant{background:var(--panel);border:1px solid var(--border)}
.msg-reasoning{background:transparent;border:1px dashed var(--border);color:var(--muted);font-style:italic;font-size:11px}
.msg-tool{background:var(--bg);border:1px solid var(--border);font-size:11px}
.msg-tool-attempt{background:rgba(234,179,8,.08);border:1px solid var(--yellow);color:var(--yellow);font-size:11px}
.msg-error{background:rgba(239,68,68,.1);border:1px solid var(--red);color:var(--red)}
.msg-role{font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:3px;
          display:flex;align-items:center;gap:6px}
.tool-badge{display:inline-block;padding:1px 6px;border-radius:3px;color:#0f1117;font-weight:700;
            font-size:9px;text-transform:none;letter-spacing:0}
.msg-body{white-space:pre-wrap;word-break:break-word}
.msg-live{border-color:#22d3ee;box-shadow:0 0 8px rgba(34,211,238,.25)}
.msg-live .msg-body::after{
  content:'\\u2588';color:#22d3ee;margin-left:1px;
  animation:alfa1-blink .8s steps(1) infinite;
}

.msg-thinking{display:flex;gap:4px;align-items:center;background:var(--panel);border:1px solid var(--border);
              align-self:flex-start;padding:10px 12px;width:fit-content}
.msg-thinking .dot{width:6px;height:6px;border-radius:50%;background:var(--muted);
                    animation:alfa1-bounce 1.1s infinite ease-in-out}
.msg-thinking .dot:nth-child(2){animation-delay:.15s}
.msg-thinking .dot:nth-child(3){animation-delay:.3s}
@keyframes alfa1-bounce{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}

#chat-input-wrap{padding:10px;border-top:1px solid var(--border);flex-shrink:0}
#chat-input{width:100%;background:var(--panel);border:1px solid var(--border);border-radius:6px;
            color:var(--text);font-family:var(--font);font-size:12px;padding:8px;resize:none}
#chat-input:focus{border-color:var(--accent);outline:none}
#chat-send{margin-top:6px;background:var(--accent);color:#fff;border:none;border-radius:5px;
           padding:6px 14px;font-size:11px;cursor:pointer;float:right}

/* ---- status bar ---- */
#status-bar{height:10px;margin:0 10px 10px 10px;border-radius:4px;background-size:20px 20px;flex-shrink:0}
#status-bar.status-idle{background:var(--border)}
#status-bar.status-working{
  background-image:repeating-linear-gradient(45deg,#eab308 0,#eab308 10px,#f59e0b 10px,#f59e0b 20px);
  animation:alfa1-stripes 1s linear infinite;
}
#status-bar.status-ok{
  background-image:repeating-linear-gradient(45deg,#22c55e 0,#22c55e 10px,#16a34a 10px,#16a34a 20px);
}
#status-bar.status-error{
  background-image:repeating-linear-gradient(45deg,#ef4444 0,#ef4444 10px,#dc2626 10px,#dc2626 20px);
}
@keyframes alfa1-stripes{from{background-position:0 0}to{background-position:20px 0}}

/* ---- workspace picker overlay ---- */
#picker-overlay{position:fixed;inset:0;background:rgba(15,17,23,.92);display:flex;
                align-items:center;justify-content:center;z-index:100}
#picker-box{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:28px 32px;text-align:center}
#picker-box h2{color:var(--accent);margin:0 0 12px}
#picker-box p{color:var(--muted);margin:0 0 18px;font-size:12px}
#picker-box button{background:var(--accent);color:#fff;border:none;border-radius:6px;padding:10px 18px;
                    font-size:12px;cursor:pointer}
.hidden{display:none !important}
"""

_ALFA1_JS = """
let workspaceRoot = null;
let openTabs = [];      // [{path, content, lang}]
let activeTab = null;
let thinkingEl = null;

/* ---------- workspace ---------- */
async function checkWorkspace(){
  const r = await fetch('/alfa1/workspace');
  const j = await r.json();
  if(j.root){ workspaceRoot = j.root; updateWorkspaceBar(); hidePicker(); loadTree(); }
  else { showPicker(); }
}
function updateWorkspaceBar(){
  const el = document.getElementById('workspace-bar');
  el.textContent = workspaceRoot || '(no folder selected)';
  el.title = workspaceRoot || '';
}
function showPicker(){ document.getElementById('picker-overlay').classList.remove('hidden'); }
function hidePicker(){ document.getElementById('picker-overlay').classList.add('hidden'); }
async function pickFolder(){
  const r = await fetch('/alfa1/workspace/pick', {method:'POST'});
  const j = await r.json();
  if(j.root){ workspaceRoot = j.root; updateWorkspaceBar(); hidePicker(); loadTree(); }
}

/* ---------- file tree ----------
 * The backend already returns the FULL recursive listing in one call (see
 * alfa1_tools.list_tree), so this builds a collapsible tree purely
 * client-side: expandedDirs tracks which folders are open, and an entry is
 * only rendered if every one of its ancestor directories is in that set —
 * no re-fetch needed on expand/collapse, only on an actual file-tree
 * change (loadTree() is called after every write/delete/patch already). */
let treeEntries = [];
let expandedDirs = new Set();

const FILE_ICONS = {
  py:'🐍', js:'📜', jsx:'📜', mjs:'📜', ts:'📘', tsx:'📘',
  css:'🎨', scss:'🎨', html:'🌐', htm:'🌐', json:'📋', yml:'⚙️', yaml:'⚙️',
  toml:'⚙️', ini:'⚙️', cfg:'⚙️', md:'📝', txt:'📝', sh:'💻', bash:'💻',
  png:'🖼️', jpg:'🖼️', jpeg:'🖼️', gif:'🖼️', svg:'🖼️', ico:'🖼️',
};
function fileIcon(path){
  const ext = path.includes('.') ? path.split('.').pop().toLowerCase() : '';
  return FILE_ICONS[ext] || '📄';
}

async function loadTree(){
  const r = await fetch('/alfa1/files/tree?path=.');
  const j = await r.json();
  treeEntries = j.entries || [];
  renderTree();
}
function toggleDir(path){
  if(expandedDirs.has(path)){
    expandedDirs.delete(path);
    // collapse descendants too, so re-expanding always starts from a clean state
    for(const p of Array.from(expandedDirs)){ if(p.startsWith(path + '/')) expandedDirs.delete(p); }
  } else {
    expandedDirs.add(path);
  }
  renderTree();
}
function renderTree(){
  const el = document.getElementById('tree');
  el.innerHTML = '';
  for(const e of treeEntries){
    const parts = e.path.split('/');
    const depth = parts.length - 1;
    const parentPath = parts.slice(0, -1).join('/');
    if(depth > 0 && !expandedDirs.has(parentPath)) continue;
    const isDir = e.type === 'dir';
    const div = document.createElement('div');
    div.className = 'tree-entry ' + (isDir ? 'dir' : 'file');
    div.style.paddingLeft = (6 + depth * 14) + 'px';
    const icon = isDir ? (expandedDirs.has(e.path) ? '📂' : '📁') : fileIcon(e.path);
    div.innerHTML = '<span class="tree-icon">' + icon + '</span>' + escapeHtml(parts[parts.length - 1]);
    div.title = e.path;
    div.onclick = isDir ? (() => toggleDir(e.path)) : (() => openFile(e.path));
    el.appendChild(div);
  }
}

/* ---------- syntax highlighter (homemade, single-pass, regex-based) ----------
 * Built from plain regex LITERALS with zero internal capturing groups (only
 * non-capturing (?:...) / lookaheads), combined at runtime via `.source` into
 * one alternation regex per language: (com)|(str)|(kw)|(num). A single
 * left-to-right scan with regex.exec() classifies and HTML-escapes each
 * matched token exactly once and copies the untouched text between matches —
 * there is no separate placeholder/restore pass, so nothing can double-match
 * or corrupt earlier substitutions. */
const LANG_RULES = {
  py:   {com:/#.*/, str:/"(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*'/,
         kw:/\\b(?:def|class|if|elif|else|for|while|return|import|from|as|with|try|except|finally|raise|pass|break|continue|lambda|yield|async|await|None|True|False|and|or|not|in|is)\\b/,
         num:/\\b\\d+\\.?\\d*\\b/},
  js:   {com:/\\/\\/.*|\\/\\*[\\s\\S]*?\\*\\//, str:/`(?:[^`\\\\]|\\\\.)*`|"(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*'/,
         kw:/\\b(?:function|const|let|var|if|else|for|while|return|import|from|export|default|class|extends|new|try|catch|finally|throw|async|await|typeof|instanceof|null|undefined|true|false)\\b/,
         num:/\\b\\d+\\.?\\d*\\b/},
  html: {com:/<!--[\\s\\S]*?-->/, str:/"(?:[^"\\\\]|\\\\.)*"/,
         kw:/<\\/?[a-zA-Z][a-zA-Z0-9-]*/, num:/(?!)/},
  css:  {com:/\\/\\*[\\s\\S]*?\\*\\//, str:/"(?:[^"\\\\]|\\\\.)*"/,
         kw:/(?:[.#]?[a-zA-Z-]+(?=\\s*\\{)|[a-z-]+(?=\\s*:))/,
         num:/\\d+\\.?\\d*(?:px|em|rem|%)?/},
  json: {com:/(?!)/, str:/"(?:[^"\\\\]|\\\\.)*"/, kw:/\\b(?:true|false|null)\\b/, num:/-?\\d+\\.?\\d*/},
  sh:   {com:/#.*/, str:/"(?:[^"\\\\]|\\\\.)*"|'[^']*'/,
         kw:/\\b(?:if|then|else|fi|for|do|done|while|function|echo|export|return)\\b/, num:/(?!)/},
};
const LANG_REGEX_CACHE = {};
function getLangRegex(lang){
  if(lang in LANG_REGEX_CACHE) return LANG_REGEX_CACHE[lang];
  const r = LANG_RULES[lang];
  if(!r){ LANG_REGEX_CACHE[lang] = null; return null; }
  const combined = new RegExp(
    '(' + r.com.source + ')|(' + r.str.source + ')|(' + r.kw.source + ')|(' + r.num.source + ')', 'g'
  );
  LANG_REGEX_CACHE[lang] = combined;
  return combined;
}
function extToLang(path){
  const ext = path.split('.').pop().toLowerCase();
  if(['py'].includes(ext)) return 'py';
  if(['js','jsx','ts','tsx','mjs'].includes(ext)) return 'js';
  if(['html','htm'].includes(ext)) return 'html';
  if(['css'].includes(ext)) return 'css';
  if(['json'].includes(ext)) return 'json';
  if(['sh','bash'].includes(ext)) return 'sh';
  return null;
}
function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function highlight(code, lang){
  const regex = getLangRegex(lang);
  if(!regex) return escapeHtml(code);
  regex.lastIndex = 0;
  let out = '', last = 0, m;
  while((m = regex.exec(code)) !== null){
    if(m.index > last) out += escapeHtml(code.slice(last, m.index));
    let cls = 'tok-num';
    if(m[1] !== undefined) cls = 'tok-com';
    else if(m[2] !== undefined) cls = 'tok-str';
    else if(m[3] !== undefined) cls = 'tok-kw';
    out += '<span class="' + cls + '">' + escapeHtml(m[0]) + '</span>';
    last = regex.lastIndex;
    if(m[0].length === 0) regex.lastIndex++;
  }
  out += escapeHtml(code.slice(last));
  return out;
}

/* ---------- open files / tabs ---------- */
async function openFile(path){
  const r = await fetch('/alfa1/files/content?path=' + encodeURIComponent(path));
  const j = await r.json();
  if(j.error){ return; }
  let tab = openTabs.find(t => t.path === path);
  if(!tab){ tab = {path, content: j.content, binary: j.binary}; openTabs.push(tab); }
  else { tab.content = j.content; tab.binary = j.binary; }
  activeTab = path;
  renderTabs();
  renderActiveFile();
}
function closeTab(path, evt){
  evt.stopPropagation();
  openTabs = openTabs.filter(t => t.path !== path);
  if(activeTab === path) activeTab = openTabs.length ? openTabs[openTabs.length-1].path : null;
  renderTabs();
  renderActiveFile();
}
function renderTabs(){
  const el = document.getElementById('tabs');
  el.innerHTML = '';
  for(const t of openTabs){
    const div = document.createElement('div');
    div.className = 'tab ' + (t.path === activeTab ? 'active' : '');
    div.textContent = t.path.split('/').pop();
    div.onclick = () => { activeTab = t.path; renderTabs(); renderActiveFile(); };
    const close = document.createElement('span');
    close.className = 'close'; close.textContent = '\\u00d7';
    close.onclick = (e) => closeTab(t.path, e);
    div.appendChild(close);
    el.appendChild(div);
  }
}
async function saveActiveFile(){
  if(!activeTab) return;
  const tab = openTabs.find(t => t.path === activeTab);
  const pre = document.getElementById('code-pre');
  if(!tab || !pre) return;
  tab.content = pre.textContent;
  await fetch('/alfa1/files/content', {
    method:'PUT', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: tab.path, content: tab.content}),
  });
}
function renderActiveFile(){
  const view = document.getElementById('code-view');
  if(!activeTab){ view.innerHTML = '<div id="code-empty">No file open</div>'; return; }
  const tab = openTabs.find(t => t.path === activeTab);
  if(!tab || tab.binary){ view.innerHTML = '<div id="code-empty">Binary file</div>'; return; }
  const lang = extToLang(tab.path);
  view.innerHTML = '<pre id="code-pre" contenteditable="true" spellcheck="false">' + highlight(tab.content || '', lang) + '</pre>';
  const pre = document.getElementById('code-pre');
  pre.addEventListener('keydown', (e) => {
    if((e.ctrlKey || e.metaKey) && e.key === 's'){ e.preventDefault(); saveActiveFile(); }
  });
}
async function openFileFresh(path){
  // Like openFile, but always refetches even if a tab is already open, and
  // reveals the new content against whatever was already shown for this
  // file — used after the agent writes/patches a file so column 2 shows
  // exactly what changed, not a full re-type of the whole file nor the
  // model's raw patch syntax (that only ever appears in the reasoning/tool
  // log on the right, never in the code pane).
  const r = await fetch('/alfa1/files/content?path=' + encodeURIComponent(path));
  const j = await r.json();
  if(j.error){ return; }
  let tab = openTabs.find(t => t.path === path);
  const oldContent = tab ? tab.content : null;
  if(!tab){ tab = {path, content: j.content, binary: j.binary}; openTabs.push(tab); }
  else { tab.content = j.content; tab.binary = j.binary; }
  activeTab = path;
  renderTabs();
  if(tab.binary){ renderActiveFile(); return; }
  revealCode(tab.content || '', extToLang(tab.path), oldContent);
}

function diffLines(oldLines, newLines){
  // Line-level LCS diff: returns newLines annotated with whether each one
  // existed unchanged (in order) in oldLines or is new/changed. Good enough
  // for the size of files this agent deals with — not trying to be a full
  // Myers diff, just enough to tell "same line" from "changed line" so the
  // reveal only animates what actually moved.
  const m = oldLines.length, n = newLines.length;
  const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));
  for(let i = m - 1; i >= 0; i--){
    for(let j = n - 1; j >= 0; j--){
      dp[i][j] = oldLines[i] === newLines[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const result = [];
  let i = 0, j = 0;
  while(i < m && j < n){
    if(oldLines[i] === newLines[j]){ result.push({text: newLines[j], changed: false}); i++; j++; }
    else if(dp[i + 1][j] >= dp[i][j + 1]){ i++; }
    else { result.push({text: newLines[j], changed: true}); j++; }
  }
  while(j < n){ result.push({text: newLines[j], changed: true}); j++; }
  return result;
}

function revealCode(code, lang, oldContent){
  // "Materializing" reveal, cyberpunk-terminal style: changed lines fade/
  // glow in with a neon drop-shadow that settles to normal and a blinking
  // cursor tracks the reveal point; UNCHANGED lines (per diffLines against
  // whatever this file showed before) appear instantly, statically, with
  // no animation — so editing one function doesn't re-type the whole file.
  // With no prior content to diff against (a brand-new file), every line
  // is treated as changed. Highlighting is done per-line (not on the whole
  // file at once like renderActiveFile does) so a construct spanning
  // multiple lines (a block comment, a multi-line string) can't leave an
  // unclosed <span> straddling two separately-inserted line elements — a
  // real risk if the combined highlighted HTML were naively split on '\\n'.
  // Purely a cosmetic simplification: only this animated reveal loses
  // cross-line highlighting fidelity, not the normal static view.
  const view = document.getElementById('code-view');
  view.innerHTML = '';
  const pre = document.createElement('pre');
  pre.id = 'code-pre';
  pre.spellcheck = false;
  view.appendChild(pre);
  view.classList.add('revealing');

  const newLines = code.split('\\n');
  const steps = oldContent != null
    ? diffLines(oldContent.split('\\n'), newLines)
    : newLines.map(text => ({text, changed: true}));
  const changedCount = steps.filter(s => s.changed).length || 1;
  const perLineMs = Math.max(3, Math.min(35, Math.floor(700 / changedCount)));
  let i = 0;

  function finish(){
    const cursor = pre.querySelector('.reveal-cursor');
    if(cursor){ cursor.remove(); }
    view.classList.remove('revealing');
    pre.contentEditable = 'true';
    pre.addEventListener('keydown', (e) => {
      if((e.ctrlKey || e.metaKey) && e.key === 's'){ e.preventDefault(); saveActiveFile(); }
    });
  }

  function step(){
    const prevCursor = pre.querySelector('.reveal-cursor');
    if(prevCursor){ prevCursor.remove(); }
    if(i >= steps.length){ finish(); return; }
    const s = steps[i];
    const div = document.createElement('div');
    div.className = s.changed ? 'code-line' : 'code-line code-line-static';
    div.innerHTML = highlight(s.text, lang) || '\\u00a0';
    if(s.changed){
      const cursor = document.createElement('span');
      cursor.className = 'reveal-cursor';
      div.appendChild(cursor);
    }
    pre.appendChild(div);
    view.scrollTop = view.scrollHeight;
    i++;
    if(i < steps.length){ setTimeout(step, s.changed ? perLineMs : 0); } else { finish(); }
  }
  step();
}

/* ---------- chat / agent ---------- */
const TOOL_COLORS = {
  write_file:'#7c6af7', read_file:'#38bdf8', delete_file:'#ef4444',
  run_command:'#eab308', list_files:'#22c55e', search_files:'#2dd4bf',
  apply_patch:'#f472b6',
};
function addMsg(role, content){
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'msg msg-' + role;
  const label = document.createElement('div');
  label.className = 'msg-role'; label.textContent = role;
  div.appendChild(label);
  const body = document.createElement('div');
  body.className = 'msg-body';
  body.textContent = content;
  div.appendChild(body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}
function renderSnapshotConversation(conversation){
  // Replays a restored session (server restart / tab reopened — see
  // alfa1_tools.save_history/load_history) into the visible chat log. Only
  // real user/assistant turns are shown; the synthetic "[alfa1] Action
  // results..." bookkeeping messages and per-turn tool_call/tool_result
  // detail aren't replayed (that granular event stream isn't persisted,
  // only the raw conversation) — this is a readable summary, not a replay.
  const log = document.getElementById('chat-log');
  log.innerHTML = '';
  for(const m of conversation){
    if(m.role !== 'user' && m.role !== 'assistant') continue;
    if(!m.content) continue;
    if(m.role === 'user' && m.content.startsWith('[alfa1]')) continue;
    addMsg(m.role, m.content);
  }
}
function addToolMsg(kind, name, detail){
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'msg msg-tool';
  const label = document.createElement('div');
  label.className = 'msg-role';
  const badge = document.createElement('span');
  badge.className = 'tool-badge';
  badge.style.background = TOOL_COLORS[name] || '#64748b';
  badge.textContent = name;
  label.appendChild(badge);
  const kindSpan = document.createElement('span');
  kindSpan.textContent = kind;
  label.appendChild(kindSpan);
  div.appendChild(label);
  const body = document.createElement('div');
  body.className = 'msg-body';
  body.textContent = detail;
  div.appendChild(body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function showThinking(){
  if(thinkingEl) return;
  const log = document.getElementById('chat-log');
  thinkingEl = document.createElement('div');
  thinkingEl.className = 'msg msg-thinking';
  thinkingEl.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  log.appendChild(thinkingEl);
  log.scrollTop = log.scrollHeight;
}
function hideThinking(){
  if(thinkingEl){ thinkingEl.remove(); thinkingEl = null; }
}
let liveReasoningEl = null, liveContentEl = null;
function appendDelta(kind, text){
  hideThinking();
  if(kind === 'reasoning'){
    if(!liveReasoningEl){ liveReasoningEl = addMsg('reasoning', ''); liveReasoningEl.classList.add('msg-live'); }
    liveReasoningEl.querySelector('.msg-body').textContent += text;
  } else {
    if(!liveContentEl){ liveContentEl = addMsg('assistant', ''); liveContentEl.classList.add('msg-live'); }
    liveContentEl.querySelector('.msg-body').textContent += text;
  }
  const log = document.getElementById('chat-log');
  log.scrollTop = log.scrollHeight;
}
function clearLiveDeltas(){
  if(liveReasoningEl){ liveReasoningEl.remove(); liveReasoningEl = null; }
  if(liveContentEl){ liveContentEl.remove(); liveContentEl = null; }
}

function setStatus(status){
  const bar = document.getElementById('status-bar');
  bar.className = 'status-' + status;
  const dot = document.getElementById('activity-dot');
  dot.className = 'activity-dot ' + (status === 'working' ? 'working' : status === 'ok' ? 'ok' : status === 'error' ? 'error' : '');
  document.getElementById('cancel-btn').classList.toggle('hidden', status !== 'working');
}
async function cancelTurn(){
  await fetch('/alfa1/cancel', {method:'POST'});
}
async function startNewTask(){
  await fetch('/alfa1/reset', {method:'POST'});
  document.getElementById('chat-log').innerHTML = '';
}
async function clearAllSessions(){
  if(!confirm('Permanently delete the stored chat history for this folder? This cannot be undone.')) return;
  await fetch('/alfa1/history', {method:'DELETE'});
  document.getElementById('chat-log').innerHTML = '';
}
async function sendMessage(){
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if(!message) return;
  addMsg('user', message);
  input.value = '';
  const r = await fetch('/alfa1/chat', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({message}),
  });
  const j = await r.json();
  if(j.error){
    addMsg('error', j.error);
    // The backend loses its workspace on restart (in-memory only, by
    // design), but a tab left open across that restart keeps showing the
    // stale tree/path — re-sync instead of leaving the user stuck on a
    // confusing error with no obvious next step.
    if(String(j.error).toLowerCase().includes('workspace')){ checkWorkspace(); }
  } else if(j.queued){
    addMsg('tool', 'Queued (position ' + j.position + ') — the agent is still busy with a previous message.');
  }
}

/* ---------- SSE: agent turn events ---------- */
let pendingWriteFilePath = null;
function connectAgentStream(){
  const es = new EventSource('/alfa1/stream');
  es.onmessage = (ev) => {
    try{
      const msg = JSON.parse(ev.data);
      if(msg.type === 'status'){ setStatus(msg.status); if(msg.status !== 'working'){ hideThinking(); clearLiveDeltas(); } }
      else if(msg.type === 'thinking'){ showThinking(); }
      else if(msg.type === 'reasoning_delta'){ appendDelta('reasoning', msg.text); }
      else if(msg.type === 'content_delta'){ appendDelta('content', msg.text); }
      else if(msg.type === 'reasoning'){ hideThinking(); clearLiveDeltas(); addMsg('reasoning', msg.content); }
      else if(msg.type === 'tool_call'){
        hideThinking(); clearLiveDeltas();
        addToolMsg('call', msg.name, JSON.stringify(msg.arguments));
        if(msg.name === 'write_file' || msg.name === 'apply_patch'){
          pendingWriteFilePath = msg.arguments && msg.arguments.path;
        }
      }
      else if(msg.type === 'tool_result'){
        addToolMsg('result', msg.name, msg.result);
        loadTree();
        // The agent writes/patches files directly (not through the PUT
        // route), so it has no file_changed event of its own — open the
        // file it just touched in column 2 using the path captured from
        // the preceding tool_call. openFileFresh diffs against whatever
        // was already shown for that file and only animates the lines that
        // actually changed — the raw patch text (<<<<<<< SEARCH etc.) is
        // never shown here, only in the reasoning/tool log on the right.
        if((msg.name === 'write_file' || msg.name === 'apply_patch') && pendingWriteFilePath){
          openFileFresh(pendingWriteFilePath);
          pendingWriteFilePath = null;
        }
      }
      else if(msg.type === 'assistant'){ hideThinking(); clearLiveDeltas(); addMsg('assistant', msg.content); }
      else if(msg.type === 'tool_attempt_unrecognized'){
        hideThinking(); clearLiveDeltas();
        const desc = msg.description || 'Model attempted an unsupported tool-call format and was asked to retry.';
        addMsg('tool-attempt', 'Unsupported tool-call format — asked to retry. ' + desc);
      }
      else if(msg.type === 'truncated'){ hideThinking(); clearLiveDeltas(); addMsg('tool-attempt', 'Reply was cut off before completing an action (hit the length limit) — asked to retry more concisely.'); }
      else if(msg.type === 'error'){ hideThinking(); clearLiveDeltas(); addMsg('error', msg.message); }
      else if(msg.type === 'file_changed'){ loadTree(); openFileFresh(msg.path); }
      else if(msg.type === 'snapshot'){ setStatus(msg.status || 'idle'); if(msg.conversation && msg.conversation.length){ renderSnapshotConversation(msg.conversation); } }
    }catch(e){}
  };
}

/* ---------- stats cards (reuse existing /v1/stats/stream) ---------- */
function connectStatsStream(){
  const es = new EventSource('/v1/stats/stream');
  es.onmessage = (ev) => {
    try{
      const msg = JSON.parse(ev.data);
      if(msg.stats){
        document.getElementById('stat-in').textContent = msg.stats.total_tokens_saved ?? 0;
        document.getElementById('stat-out').textContent = msg.stats.total_output_tokens_saved ?? 0;
      }
    }catch(e){}
  };
}

/* ---------- resizable columns ---------- */
function setupResizer(id, colEl){
  const handle = document.getElementById(id);
  let dragging = false;
  handle.addEventListener('mousedown', () => { dragging = true; handle.classList.add('active'); });
  window.addEventListener('mouseup', () => { dragging = false; handle.classList.remove('active'); });
  window.addEventListener('mousemove', (e) => {
    if(!dragging) return;
    const rect = colEl.getBoundingClientRect();
    const w = Math.max(160, e.clientX - rect.left);
    colEl.style.width = w + 'px';
  });
}

/* ---------- init ---------- */
document.getElementById('pick-btn').onclick = pickFolder;
document.getElementById('picker-box-btn').onclick = pickFolder;
document.getElementById('chat-send').onclick = sendMessage;
document.getElementById('cancel-btn').onclick = cancelTurn;
document.getElementById('new-task-btn').onclick = startNewTask;
document.getElementById('clear-sessions-btn').onclick = clearAllSessions;
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }
});
setupResizer('resizer-1', document.getElementById('col1'));
setupResizer('resizer-2', document.getElementById('col2'));
setStatus('idle');
checkWorkspace();
connectAgentStream();
connectStatsStream();
"""

ALFA1_HTML = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alfa1</title>
<style>{_ALFA1_CSS}</style>
</head>
<body>
<div id="layout">
  <div id="col1">
    <div id="tree-header"><h1>ALFA1</h1><button id="pick-btn">Folder</button></div>
    <div id="workspace-bar">(no folder selected)</div>
    <div id="tree"></div>
    <div id="stats-cards">
      <div class="stat-card"><div class="label">Input tokens saved</div><div class="value" id="stat-in">0</div></div>
      <div class="stat-card"><div class="label">Output tokens saved</div><div class="value" id="stat-out">0</div></div>
    </div>
  </div>
  <div class="resizer" id="resizer-1"></div>
  <div id="col2">
    <div id="tabs"></div>
    <div id="code-view"><div id="code-empty">No file open</div></div>
  </div>
  <div class="resizer" id="resizer-2"></div>
  <div id="col3">
    <div id="col3-topbar">
      <span class="title">Activity Status</span>
      <button id="cancel-btn" class="hidden">Cancel</button>
      <button id="new-task-btn" class="topbar-icon-btn" title="Start a new task (clears the current chat)">+ New Task</button>
      <button id="clear-sessions-btn" class="topbar-icon-btn" title="Permanently delete the stored chat history for this folder">Clear Sessions</button>
      <span class="activity-dot" id="activity-dot"></span>
    </div>
    <div id="chat-log"></div>
    <div id="chat-input-wrap">
      <textarea id="chat-input" rows="3" placeholder="Ask Alfa1 to build something... (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="chat-send">Send</button>
    </div>
    <div id="status-bar" class="status-idle"></div>
  </div>
</div>
<div id="picker-overlay" class="hidden">
  <div id="picker-box">
    <h2>Select a workspace folder</h2>
    <p>Alfa1 needs a folder to work in. It will have full read/write/execute permissions inside it.</p>
    <button id="picker-box-btn">Select Folder</button>
  </div>
</div>
<script>{_ALFA1_JS}</script>
</body>
</html>
"""
