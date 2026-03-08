"""
app.py
------
Streamlit frontend for CAMS Voice & Chat Assistant.
Uses a full-page HTML/JS component for the chat UI so the input bar
(mic + upload + text) is TRULY fixed at the bottom at all times.
FastAPI backend is called directly from JS via fetch().
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript

st.set_page_config(
    page_title="CAMS Voice & Chat Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all Streamlit chrome
st.markdown("""
<style>
    #MainMenu, header, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { visibility: hidden !important; height: 0 !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    section.main > div { padding: 0 !important; }
    section.main { overflow: hidden !important; }
    iframe { display: block !important; border: none !important; }
    [data-testid="stAppViewContainer"] { overflow: hidden !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Get real browser window height
win_height = st_javascript("window.innerHeight")
# Subtract ~10px for any residual Streamlit chrome; fallback to 800
iframe_height = int(win_height) - 80 if win_height and int(win_height) > 100 else 720

API_BASE = "http://localhost:8000"

components.html(f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CAMS Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0d0f14;
    --surface:   #161922;
    --surface2:  #1e2230;
    --border:    #2a2f3d;
    --accent:    #4f8ef7;
    --accent2:   #e05a5a;
    --green:     #3ecf78;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --user-bg:   #1a2540;
    --bot-bg:    #161922;
    --radius:    12px;
    --font:      'IBM Plex Sans', sans-serif;
  }}

  #lang-badge {{
    display: none;
  }}
  /* JS will set iframe height — this ensures html/body fill it */
  html, body {{
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    overflow: hidden;
  }}

  /* ── Layout ── */
  #app {{
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 860px;
    margin: 0 auto;
  }}

  /* ── Header ── */
  #header {{
    padding: 16px 20px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
    background: var(--bg);
  }}
  #header .logo {{
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #e05a5a, #f59e0b);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }}
  #header h1 {{ font-size: 16px; font-weight: 600; letter-spacing: -0.3px; }}
  #header p  {{ font-size: 11px; color: var(--muted); margin-top: 1px; }}
  #lang-badge {{
    margin-left: auto;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: var(--muted);
  }}
  #lang-badge span {{ color: var(--accent); font-weight: 600; }}

  /* ── Chat window ── */
  #chat {{
    flex: 1;
    overflow-y: auto;
    padding: 20px 20px 8px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }}
  #chat::-webkit-scrollbar {{ width: 4px; }}
  #chat::-webkit-scrollbar-track {{ background: transparent; }}
  #chat::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

  /* ── Messages ── */
  .msg {{ display: flex; gap: 10px; animation: fadeIn .25s ease; }}
  @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform: none; }} }}

  .msg.user  {{ flex-direction: row-reverse; }}
  .avatar {{
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 15px;
  }}
  .msg.user  .avatar {{ background: linear-gradient(135deg,#4f8ef7,#7c3aed); }}
  .msg.bot   .avatar {{ background: linear-gradient(135deg,#e05a5a,#f59e0b); }}

  .bubble {{
    max-width: 75%;
    padding: 12px 15px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 13.5px;
  }}
  .msg.user .bubble {{
    background: var(--user-bg);
    border: 1px solid #2a3a60;
    border-top-right-radius: 3px;
  }}
  .msg.bot .bubble {{
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 3px;
    width: 100%;
    max-width: 100%;
  }}

  /* ── Bot response inner ── */
  .response-text {{ margin-bottom: 12px; line-height: 1.65; }}

  .metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin: 10px 0;
  }}
  .metric {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .metric .label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }}
  .metric .value {{ font-size: 13px; font-weight: 600; }}
  .metric .sub   {{ font-size: 10px; color: var(--muted); margin-top: 2px; }}

  .transcript-box {{
    margin-top: 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--muted);
  }}
  .transcript-box strong {{ color: var(--text); display: block; margin-bottom: 2px; font-size: 11px; }}

  audio {{
    width: 100%;
    height: 36px;
    border-radius: 8px;
    margin-bottom: 10px;
    accent-color: var(--accent);
  }}

  /* ── Typing indicator ── */
  .typing {{ display: flex; gap: 4px; padding: 6px 0; align-items: center; }}
  .typing span {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted);
    animation: bounce 1.2s infinite;
  }}
  .typing span:nth-child(2) {{ animation-delay: .2s; }}
  .typing span:nth-child(3) {{ animation-delay: .4s; }}
  @keyframes bounce {{
    0%,60%,100% {{ transform: translateY(0); }}
    30% {{ transform: translateY(-6px); background: var(--accent); }}
  }}

  /* ── INPUT BAR — truly fixed at bottom ── */
  #input-bar {{
    flex-shrink: 0;
    border-top: 1px solid var(--border);
    background: var(--bg);
    padding: 10px 16px 14px;
  }}

  #toolbar {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }}

  /* Mic button */
  #mic-btn {{
    width: 38px; height: 38px;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    transition: all .2s;
    flex-shrink: 0;
    position: relative;
  }}
  #mic-btn:hover {{ border-color: var(--accent); background: var(--surface2); }}
  #mic-btn.recording {{
    background: var(--accent2);
    border-color: var(--accent2);
    animation: pulse-ring 1.2s infinite;
  }}
  @keyframes pulse-ring {{
    0%   {{ box-shadow: 0 0 0 0 rgba(224,90,90,.5); }}
    70%  {{ box-shadow: 0 0 0 10px rgba(224,90,90,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(224,90,90,0); }}
  }}

  #mic-label {{
    font-size: 10px; color: var(--muted);
    white-space: nowrap;
  }}
  #mic-status {{
    font-size: 10px;
    color: var(--accent2);
    display: none;
    animation: blink 1s infinite;
  }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}

  /* File upload */
  #file-label {{
    display: flex; align-items: center; gap: 6px;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 0 12px;
    height: 38px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text);
    transition: all .2s;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  #file-label:hover {{ border-color: var(--accent); background: var(--surface2); }}
  #file-input {{ display: none; }}
  #file-name {{ font-size: 11px; color: var(--muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  #send-audio-btn {{
    height: 38px; padding: 0 14px;
    background: var(--accent);
    color: white;
    border: none; border-radius: 8px;
    font-size: 12px; font-weight: 600;
    cursor: pointer;
    display: none;
    transition: all .2s;
    flex-shrink: 0;
  }}
  #send-audio-btn:hover {{ background: #3a7ce0; }}

  /* Text input row */
  #text-row {{
    display: flex; gap: 8px; align-items: center;
  }}
  #text-input {{
    flex: 1;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    color: var(--text);
    font-family: var(--font);
    font-size: 13.5px;
    outline: none;
    transition: border-color .2s;
    resize: none;
    height: 42px;
    line-height: 1.4;
  }}
  #text-input:focus {{ border-color: var(--accent); }}
  #text-input::placeholder {{ color: var(--muted); }}

  #send-btn {{
    width: 42px; height: 42px;
    background: var(--accent);
    border: none; border-radius: 10px;
    color: white; font-size: 18px;
    cursor: pointer; flex-shrink: 0;
    transition: all .2s;
    display: flex; align-items: center; justify-content: center;
  }}
  #send-btn:hover {{ background: #3a7ce0; transform: scale(1.05); }}
  #send-btn:disabled {{ background: var(--surface2); cursor: not-allowed; transform: none; }}

  .escalation-alert {{
    background: rgba(224,90,90,.12);
    border: 1px solid var(--accent2);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--accent2);
    margin-bottom: 10px;
  }}
</style>
</head>
<body>
<div id="app">

  <!-- Header -->
  <div id="header">
    <div class="logo">🎙️</div>
    <div>
      <h1>CAMS Voice &amp; Chat Assistant</h1>
      <p>Mutual fund queries in English, Hindi &amp; Tamil</p>
    </div>
  </div>

  <!-- Chat window -->
  <div id="chat">
    <div class="msg bot">
      <div class="avatar">🤖</div>
      <div class="bubble">
        <div class="response-text">Hello! I'm your CAMS mutual fund assistant. How can I help you today? You can type, record your voice, or upload an audio file.</div>
      </div>
    </div>
  </div>

  <!-- Fixed input bar -->
  <div id="input-bar">
    <!-- Toolbar: mic + file upload -->
    <div id="toolbar">
      <button id="mic-btn" title="Record voice">🎤</button>
      <span id="mic-label">Record</span>
      <span id="mic-status">● Recording...</span>

      <label id="file-label" title="Upload audio file">
        📁 Browse audio
        <input type="file" id="file-input" accept=".wav,.mp3,.ogg,.webm">
      </label>
      <span id="file-name"></span>
      <button id="send-audio-btn">🚀 Send Audio</button>
    </div>

    <!-- Text input row -->
    <div id="text-row">
      <input
        type="text"
        id="text-input"
        placeholder="Type your query... e.g. I want to check my SIP status"
        autocomplete="off"
      />
      <button id="send-btn" title="Send">➤</button>
    </div>
  </div>

</div>

<script>
const API_BASE    = "{API_BASE}";
const sessionId   = crypto.randomUUID();
const chat        = document.getElementById("chat");
const textInput   = document.getElementById("text-input");
const sendBtn     = document.getElementById("send-btn");
const micBtn      = document.getElementById("mic-btn");
const micStatus   = document.getElementById("mic-status");
const fileInput   = document.getElementById("file-input");
const fileName    = document.getElementById("file-name");
const sendAudio   = document.getElementById("send-audio-btn");

let mediaRecorder   = null;
let audioChunks     = [];
let isRecording     = false;
let pendingAudioBlob = null;

// ── Chat history for context ──────────────────────────────────────────────────
let conversationHistory = [];
let sessionInvestorId   = null;   // Active PAN for this session

// ── Extract PAN from text ─────────────────────────────────────────────────────
function extractPAN(text) {{
  const match = text && text.match(/\b[A-Z]{{5}}[0-9]{{4}}[A-Z]\b/i);
  return match ? match[0].toUpperCase() : null;
}}

// ── Update session PAN — new PAN always wins (user switching accounts) ────────
function updateSessionPAN(text) {{
  const pan = extractPAN(text);
  if (pan && pan !== sessionInvestorId) {{
    sessionInvestorId = pan;
    console.log("Session PAN updated:", pan);
  }}
}}

// ── Scroll to bottom ──────────────────────────────────────────────────────────
function scrollBottom() {{
  chat.scrollTop = chat.scrollHeight;
}}

// ── Add user bubble ───────────────────────────────────────────────────────────
function addUserMsg(text) {{
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="avatar">👤</div><div class="bubble">${{escHtml(text)}}</div>`;
  chat.appendChild(div);
  scrollBottom();
}}

// ── Add typing indicator ──────────────────────────────────────────────────────
function addTyping() {{
  const div = document.createElement("div");
  div.className = "msg bot";
  div.id = "typing";
  div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(div);
  scrollBottom();
  return div;
}}

// ── Remove typing indicator ───────────────────────────────────────────────────
function removeTyping() {{
  const t = document.getElementById("typing");
  if (t) t.remove();
}}

// ── Escape HTML ───────────────────────────────────────────────────────────────
function escHtml(s) {{
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

// ── Render bot response ───────────────────────────────────────────────────────
function addBotResponse(data) {{
  removeTyping();
  const div = document.createElement("div");
  div.className = "msg bot";

  const intentMap = {{
    redemption_request:"💰 Redemption Request", account_statement:"📄 Account Statement",
    compliance_query:"⚖️ Compliance Query",     portfolio_enquiry:"📊 Portfolio Enquiry",
    transaction_status:"🔄 Transaction Status", kyc_update:"🪪 KYC Update",
    dividend_info:"💵 Dividend Info",            general_enquiry:"❓ General Enquiry",
    escalation:"🚨 Escalation",                  unknown:"❔ Unknown",
  }};
  const sentMap = {{
    positive:"😊 Positive", neutral:"😐 Neutral", negative:"😞 Negative",
    frustrated:"😠 Frustrated", urgent:"⚡ Urgent",
  }};
  const actionColor = {{
    success:"🟢", pending:"🟡", blocked_compliance:"🔴", failed:"🔴", not_required:"⚪"
  }};

  const action = data.action_result || {{}};
  const status = action.status || "not_required";
  const intent = data.intent || "unknown";
  const sentiment = data.sentiment || "neutral";
  const confidence = ((data.confidence || 0) * 100).toFixed(0);

  // Update language badge

  let audioHtml = "";
  if (data.audio_url) {{
    audioHtml = `<audio autoplay controls><source src="data:audio/wav;base64,${{data.audio_url}}" type="audio/wav"></audio>`;
  }}

  let transcriptHtml = "";
  if (data.transcribed_text) {{
    transcriptHtml = `
      <div class="transcript-box">
        <strong>📝 Transcription ${{data.detected_language ? `· <code>${{data.detected_language}}</code>` : ""}}</strong>
        ${{escHtml(data.transcribed_text)}}
      </div>`;
  }}

  let escalationHtml = "";
  if (data.requires_escalation) {{
    escalationHtml = `<div class="escalation-alert">🚨 Escalated to a human agent. A representative will contact you shortly.</div>`;
  }}

  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble">
      ${{escalationHtml}}
      ${{audioHtml}}
      <div class="response-text">${{escHtml(data.response_text || "")}}</div>
      <div class="metrics">
        <div class="metric">
          <div class="label">🎯 Intent</div>
          <div class="value">${{intentMap[intent] || intent}}</div>
          <div class="sub">Confidence: ${{confidence}}%</div>
        </div>
        <div class="metric">
          <div class="label">💭 Sentiment</div>
          <div class="value">${{sentMap[sentiment] || sentiment}}</div>
        </div>
        <div class="metric">
          <div class="label">⚡ Action</div>
          <div class="value">${{actionColor[status] || "⚪"}} ${{status.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}}</div>
          ${{action.reference_id ? `<div class="sub">Ref: ${{action.reference_id}}</div>` : ""}}
        </div>
      </div>
      ${{transcriptHtml}}
    </div>`;

  chat.appendChild(div);
  scrollBottom();
}}

// ── Error bubble ──────────────────────────────────────────────────────────────
function addError(msg) {{
  removeTyping();
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = `<div class="avatar">🤖</div><div class="bubble" style="color:#e05a5a">❌ ${{escHtml(msg)}}</div>`;
  chat.appendChild(div);
  scrollBottom();
}}

// ── Send text ─────────────────────────────────────────────────────────────────
async function sendText() {{
  const msg = textInput.value.trim();
  if (!msg) return;
  addUserMsg(msg);
  textInput.value = "";
  sendBtn.disabled = true;
  addTyping();

  // Add to history before sending
  conversationHistory.push({{ role: "user", content: msg }});

  // New PAN in message always overrides session PAN (account switching)
  updateSessionPAN(msg);

  try {{
    const r = await fetch(`${{API_BASE}}/api/chat`, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{
        message:              msg,
        language:             "en-IN",
        session_id:           sessionId,
        investor_id:          sessionInvestorId || "",
        conversation_history: conversationHistory.slice(-8),
      }})
    }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "API error");

    // If LLM identified a new investor from entities, update session
    if (data.entities && data.entities.investor_id)
      updateSessionPAN(data.entities.investor_id);

    // Add assistant reply to history
    conversationHistory.push({{ role: "assistant", content: data.response_text || "" }});
    addBotResponse(data);
  }} catch(e) {{
    addError(e.message);
    conversationHistory.pop();
  }} finally {{
    sendBtn.disabled = false;
  }}
}}

// ── Send audio blob ───────────────────────────────────────────────────────────
async function sendAudioBlob(blob, filename) {{
  addUserMsg(`🎙️ ${{filename}}`);
  addTyping();
  try {{
    const form = new FormData();
    form.append("audio_file",  blob, filename);
    form.append("session_id",  sessionId);
    form.append("investor_id", sessionInvestorId || "");
    const r = await fetch(`${{API_BASE}}/api/voice`, {{ method: "POST", body: form }});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "API error");

    // Track in history so follow-up text messages have context
    if (data.transcribed_text) {{
      updateSessionPAN(data.transcribed_text);  // Extract PAN if user spoke it
      conversationHistory.push({{ role: "user",      content: data.transcribed_text }});
      conversationHistory.push({{ role: "assistant", content: data.response_text || "" }});
    }}
    if (data.entities && data.entities.investor_id)
      updateSessionPAN(data.entities.investor_id);
    addBotResponse(data);
  }} catch(e) {{
    addError(e.message);
  }}
}}

// ── Mic recording ─────────────────────────────────────────────────────────────
micBtn.addEventListener("click", async () => {{
  if (isRecording) {{
    // Stop
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove("recording");
    micBtn.textContent = "🎤";
    micStatus.style.display = "none";
  }} else {{
    // Start
    try {{
      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {{
        const blob = new Blob(audioChunks, {{ type: "audio/webm" }});
        stream.getTracks().forEach(t => t.stop());
        await sendAudioBlob(blob, "mic_recording.webm");
      }};
      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      micBtn.textContent = "⏹";
      micStatus.style.display = "inline";
    }} catch(e) {{
      addError("Microphone access denied. Please allow mic permissions.");
    }}
  }}
}});

// ── File upload ───────────────────────────────────────────────────────────────
fileInput.addEventListener("change", () => {{
  const file = fileInput.files[0];
  if (!file) return;
  pendingAudioBlob = file;
  fileName.textContent = file.name;
  sendAudio.style.display = "block";
}});

sendAudio.addEventListener("click", async () => {{
  if (!pendingAudioBlob) return;
  sendAudio.style.display = "none";
  fileName.textContent = "";
  await sendAudioBlob(pendingAudioBlob, pendingAudioBlob.name);
  pendingAudioBlob = null;
  fileInput.value = "";
}});

// ── Text input events ─────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendText);
textInput.addEventListener("keydown", e => {{
  if (e.key === "Enter" && !e.shiftKey) {{ e.preventDefault(); sendText(); }}
}});

// ── Fill full viewport height ─────────────────────────────────────────────────
function fitHeight() {{
  const h = window.innerHeight;
  document.getElementById("app").style.height = h + "px";
  document.documentElement.style.height = h + "px";
  document.body.style.height = h + "px";
  // Tell Streamlit iframe to resize
  window.parent.postMessage({{type:"streamlit:setFrameHeight", height: h}}, "*");
}}
fitHeight();
window.addEventListener("resize", fitHeight);
</script>
</body>
</html>
""", height=iframe_height, scrolling=False)