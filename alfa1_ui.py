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

/* ---- layout / columns ---- */
#layout{display:flex;height:100vh}
#col1{width:260px;min-width:160px;max-width:600px;background:var(--panel);border-right:1px solid var(--border);
      display:flex;flex-direction:column}
#col2{width:400px;min-width:200px;max-width:900px;border-right:1px solid var(--border);
      display:flex;flex-direction:column}
#col3{flex:1;min-width:280px;display:flex;flex-direction:column}
.resizer{width:5px;cursor:col-resize;background:var(--border);flex-shrink:0}
.resizer:hover,.resizer.active{background:var(--accent)}

/* ---- column 1: tree + stats ---- */
#tree-header{padding:10px 12px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
#tree-header h1{font-size:13px;color:var(--accent);margin:0;letter-spacing:.5px}
#pick-btn{background:var(--accent);color:#fff;border:none;border-radius:4px;padding:5px 8px;font-size:10px;cursor:pointer}
#workspace-bar{padding:5px 12px;font-size:9px;color:var(--muted);border-bottom:1px solid var(--border);
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
#tree{flex:1;overflow:auto;padding:6px}
.tree-entry{padding:3px 6px;border-radius:3px;cursor:pointer;white-space:nowrap;font-size:11px;color:var(--text)}
.tree-entry:hover{background:rgba(124,106,247,.12)}
.tree-entry.dir{color:var(--muted);font-weight:600}
.tree-entry.active{background:rgba(124,106,247,.25)}
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

/* ---------- file tree ---------- */
async function loadTree(){
  const r = await fetch('/alfa1/files/tree?path=.');
  const j = await r.json();
  renderTree(j.entries || []);
}
function renderTree(entries){
  const el = document.getElementById('tree');
  el.innerHTML = '';
  for(const e of entries){
    const depth = e.path.split('/').length - 1;
    const div = document.createElement('div');
    div.className = 'tree-entry ' + (e.type === 'dir' ? 'dir' : 'file');
    div.style.paddingLeft = (6 + depth*14) + 'px';
    div.textContent = (e.type === 'dir' ? '\\u25b8 ' : '') + e.path.split('/').pop();
    div.title = e.path;
    if(e.type === 'file'){ div.onclick = () => openFile(e.path); }
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
  // animates the new content materializing in rather than snapping to it
  // instantly — used after the agent writes/patches a file so column 2
  // visibly shows the change happening, not just the end result.
  const r = await fetch('/alfa1/files/content?path=' + encodeURIComponent(path));
  const j = await r.json();
  if(j.error){ return; }
  let tab = openTabs.find(t => t.path === path);
  if(!tab){ tab = {path, content: j.content, binary: j.binary}; openTabs.push(tab); }
  else { tab.content = j.content; tab.binary = j.binary; }
  activeTab = path;
  renderTabs();
  if(tab.binary){ renderActiveFile(); return; }
  revealCode(tab.content || '', extToLang(tab.path));
}

function revealCode(code, lang){
  // Line-by-line "materializing" reveal, cyberpunk-terminal style: each
  // line fades/glows in with a neon drop-shadow that settles to normal, a
  // scanline sweeps down the pane while it runs, and a blinking cursor
  // tracks the reveal point. Highlighting is done per-line (not on the
  // whole file at once like renderActiveFile does) so a construct spanning
  // multiple lines (a block comment, a multi-line string) can't leave an
  // unclosed <span> straddling two separately-inserted line elements —
  // a real risk if the combined highlighted HTML were naively split on
  // '\\n'. Purely a cosmetic simplification: only this animated reveal
  // loses cross-line highlighting fidelity, not the normal static view.
  const view = document.getElementById('code-view');
  view.innerHTML = '';
  const pre = document.createElement('pre');
  pre.id = 'code-pre';
  pre.spellcheck = false;
  view.appendChild(pre);
  view.classList.add('revealing');

  const lines = code.split('\\n');
  const total = lines.length || 1;
  const perLineMs = Math.max(3, Math.min(35, Math.floor(700 / total)));
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
    if(i >= lines.length){ finish(); return; }
    const div = document.createElement('div');
    div.className = 'code-line';
    div.innerHTML = highlight(lines[i], lang) || '\\u00a0';
    const cursor = document.createElement('span');
    cursor.className = 'reveal-cursor';
    div.appendChild(cursor);
    pre.appendChild(div);
    view.scrollTop = view.scrollHeight;
    i++;
    if(i < lines.length){ setTimeout(step, perLineMs); } else { finish(); }
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
  }
}

/* ---------- SSE: agent turn events ---------- */
let pendingWriteFilePath = null;
function connectAgentStream(){
  const es = new EventSource('/alfa1/stream');
  es.onmessage = (ev) => {
    try{
      const msg = JSON.parse(ev.data);
      if(msg.type === 'status'){ setStatus(msg.status); if(msg.status !== 'working') hideThinking(); }
      else if(msg.type === 'thinking'){ showThinking(); }
      else if(msg.type === 'reasoning'){ hideThinking(); addMsg('reasoning', msg.content); }
      else if(msg.type === 'tool_call'){
        hideThinking();
        addToolMsg('call', msg.name, JSON.stringify(msg.arguments));
        if(msg.name === 'write_file'){ pendingWriteFilePath = msg.arguments && msg.arguments.path; }
      }
      else if(msg.type === 'tool_result'){
        addToolMsg('result', msg.name, msg.result);
        loadTree();
        // The agent writes files directly (not through the PUT route), so it
        // has no file_changed event of its own — open the file it just wrote
        // in column 2 using the path captured from the preceding tool_call.
        if(msg.name === 'write_file' && pendingWriteFilePath){
          openFileFresh(pendingWriteFilePath);
          pendingWriteFilePath = null;
        }
      }
      else if(msg.type === 'assistant'){ hideThinking(); addMsg('assistant', msg.content); }
      else if(msg.type === 'tool_attempt_unrecognized'){
        hideThinking();
        const desc = msg.description || 'Model attempted an unsupported tool-call format and was asked to retry.';
        addMsg('tool-attempt', 'Unsupported tool-call format — asked to retry. ' + desc);
      }
      else if(msg.type === 'truncated'){ hideThinking(); addMsg('tool-attempt', 'Reply was cut off before completing an action (hit the length limit) — asked to retry more concisely.'); }
      else if(msg.type === 'error'){ hideThinking(); addMsg('error', msg.message); }
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
    <div id="col3-topbar"><span class="title">Activity Status</span><button id="cancel-btn" class="hidden">Cancel</button><span class="activity-dot" id="activity-dot"></span></div>
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
