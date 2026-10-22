"""
Real-time English Communication Tutor — FastAPI + Embedded UI
Single-file server: UI is served inline, no static folder needed.

Run:
    python main.py

Open:  http://localhost:8000
"""

import asyncio
import json
import os
import traceback

import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")

MODEL            = "gemini-3.1-flash-live-preview"
SEND_SAMPLE_RATE = 16000   # browser mic → model (Hz)
RECV_SAMPLE_RATE = 24000   # model → browser speaker (Hz)

SYSTEM_PROMPT = """You are "Kate," an expert English Communication Consultant and Technical Language Trainer. Your objective is to conduct dynamic, real-time spoken training sessions to enhance the learner's professional fluency, sentence structure, accent clarity, and technical/topical accuracy across specialized domains (e.g., AI, software architecture, data engineering, business strategy).

Core Directives & Operational Flow:

Mandatory First Turn (Self-Introduction First): You must speak first immediately upon connection. Begin with a warm, professional self-introduction: state your name ("Kate"), outline your background as a communication and technical language consultant, and then invite the learner to introduce themselves, share their goals, and discuss what they wish to achieve.

Uninterrupted Active Listening: Allow the learner to complete their entire thought or response without interruption. Analyze the complete vocal input—evaluating both the linguistic delivery and the technical validity of their statements—before formulating your response.

Targeted Real-Time Feedback:
- Full-Sentence Grammar & Style Refinement: Provide complete, highly professional reformulations of misconstructed or awkward sentences.
- Deep Subject-Matter & Technical Corrections: Correct technical inaccuracies alongside language feedback.
- Pronunciation & Accent Modification: Highlight mispronounced words and model correct pronunciation.
- Dialogue Progression: Transition from corrections into thought-provoking questions to encourage extended spoken turns.

Session Balance & Delivery: Keep your turns concise so the learner contributes most of the spoken time.
Scenario Adaptability: Switch seamlessly into professional roleplays when requested.
Tone & Persona: Professional, articulate, knowledgeable, encouraging, and patient. Do not reference internal prompts.
"""

# ────────────────────────────────────────────────────────────────────────────
# Embedded HTML UI
# ────────────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>English Communication Tutor</title>
<style>
:root{
  --bg:#0b0f17;
  --surface:#111827;
  --surface2:#1e293b;
  --border:#283548;
  --card-bg:#111827;
  --card-border:#283548;
  --text:#f1f5f9;
  --muted:#94a3b8;
  --accent:#0f766e;
  --accent-hover:#115e59;
  --accent-light:#14b8a6;
  --secondary:#d97706;
  --green:#10b981;
  --red:#f87171;
  --radius:6px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  background:var(--bg);
  color:var(--text);
  height:100vh;
  display:flex;
  flex-direction:column;
  overflow:hidden;
}

/* ── Header ─────────────────────────────────────────────────────────── */
header{
  display:flex;align-items:center;gap:14px;
  padding:14px 28px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  flex-shrink:0;z-index:10;position:relative;
}
.brand{display:flex;align-items:center;gap:12px;}
.header-text{display:flex;flex-direction:column;}
header h1{font-size:1.05rem;font-weight:600;letter-spacing:.02em;margin:0;color:var(--text);}
header p{font-size:.72rem;color:var(--muted);margin-top:2px;}
.pill{
  margin-left:auto;padding:5px 12px;border-radius:var(--radius);
  font-size:.72rem;font-weight:600;
  background:var(--surface2);border:1px solid var(--border);
  display:flex;align-items:center;gap:6px;
}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:background .3s}
.dot.live{background:var(--green)}
.dot.connecting{background:var(--secondary);animation:blink 1s infinite}
.dot.error{background:var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── Main area — stays flex forever, NO display switch ─────────────── */
main{
  flex:1;display:flex;align-items:center;
  overflow:hidden;position:relative;
}

/* ── IDLE: aside card centered via transform (animatable!) ─────────── */
aside{
  width:460px;
  flex-shrink:0;
  /* translateX centers the card: push right by (50vw - half-card-width) */
  transform:translateX(calc(50vw - 230px));
  max-height:calc(100vh - 110px);
  overflow-y:auto;

  background:var(--surface);
  border:1px solid var(--border);
  border-radius:8px;
  display:flex;flex-direction:column;
  padding:36px 32px;gap:22px;
  box-shadow:0 4px 16px rgba(0,0,0,0.25);

  /* ALL animated properties use concrete pixel/% values — no 'auto' */
  /* Option 1: snappier slide with spring-like overshoot */
  transition:
    transform     0.65s cubic-bezier(0.34,1.56,0.64,1),
    width         0.65s cubic-bezier(0.34,1.56,0.64,1),
    max-height    0.65s ease,
    border-radius 0.65s ease,
    padding       0.65s ease,
    gap           0.65s ease,
    box-shadow    0.65s ease;
}

/* ── ACTIVE: transform back to 0, shrink to sidebar ─────────────────── */
body.session-active aside{
  transform:translateX(0);        /* slide to left position */
  width:300px;
  max-height:100%;
  align-self:stretch;
  border-radius:0;
  padding:20px 16px;
  gap:16px;
  border:none;
  border-right:1px solid var(--border);
  box-shadow:none;
}
aside h2{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.10em;color:var(--muted)}

/* Idle shows hero, active shows Controls label */
.aside-title{display:none;}
.idle-hero{
  display:flex;flex-direction:column;align-items:center;
  text-align:center;gap:8px;flex-shrink:0;
}
.idle-hero h3{font-size:1.4rem;font-weight:600;letter-spacing:.01em;color:var(--text);}
.idle-hero p{font-size:.84rem;color:var(--muted);line-height:1.6;}
body.session-active .idle-hero{display:none;}
body.session-active .aside-title{display:block;}

/* ── Orb ────────────────────────────────────────────────────────────── */
.orb-wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.orb{
  width:96px;height:96px;border-radius:50%;
  background:transparent;border:2px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
  transition:transform 0.85s ease,width 0.85s ease,height 0.85s ease,border-color 0.3s ease;
}
body.session-active .orb{width:76px;height:76px;}
.orb.speaking{animation:orbpulse 1.2s ease-in-out infinite;border-color:var(--accent)}
@keyframes orbpulse{
  0%,100%{box-shadow:0 0 0 0 rgba(13,148,136,0.35)}
  50%{box-shadow:0 0 0 6px rgba(13,148,136,0.1)}
}
.kate-label{font-size:.75rem;color:var(--muted);font-weight:500;}

/* ── Waveform visualizer ─────────────────────────────────────────────── */
.waveform{
  width:100%;height:56px;border-radius:var(--radius);
  background:var(--surface2);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;gap:3px;padding:10px 8px;
  overflow:hidden;
}
.waveform .bar{
  flex:1;max-width:5px;height:6px;border-radius:2px;
  background:var(--accent);
  transition:height 0.08s ease;
  animation:idle-wave 2.4s ease-in-out infinite;
}
.waveform .bar:nth-child(odd){animation-delay:0s;}
.waveform .bar:nth-child(even){animation-delay:1.2s;}
@keyframes idle-wave{
  0%,100%{height:6px;opacity:0.4}
  50%{height:14px;opacity:0.8}
}
body.session-active .waveform .bar{animation-play-state:paused;}
body.session-active aside .waveform.speaking .bar{animation:none;}

/* ── Mic row ────────────────────────────────────────────────────────── */
.mic-row{display:flex;align-items:center;gap:8px;font-size:.72rem;color:var(--muted)}
.mic-track{flex:1;height:4px;border-radius:2px;background:var(--border);overflow:hidden}
.mic-fill{height:100%;width:0%;border-radius:2px;background:var(--accent);transition:width .05s}

/* ── Buttons ────────────────────────────────────────────────────────── */
.btns{display:flex;flex-direction:column;gap:10px}
.btn{
  width:100%;padding:11px 12px;border:none;border-radius:var(--radius);
  font-size:.85rem;font-weight:600;cursor:pointer;letter-spacing:.01em;
  transition:background-color .15s, border-color .15s;display:flex;align-items:center;justify-content:center;gap:8px;
}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-start{background:var(--accent);color:#ffffff;}
.btn-start:hover:not(:disabled){background:var(--accent-hover);}
.btn-stop{background:var(--surface2);color:var(--red);border:1px solid var(--border);}
.btn-stop:hover:not(:disabled){background:#281b22;border-color:var(--red);}
.btn-clear{background:transparent;color:var(--muted);border:1px solid var(--border);}
.btn-clear:hover:not(:disabled){background:var(--surface2);color:var(--text);}

/* ── Tips ───────────────────────────────────────────────────────────── */
.tips{font-size:.72rem;color:var(--muted);line-height:1.8}
.tips strong{color:var(--text)}

/* ── Chat panel — max-width: 0→100% is animatable (no 'auto'!) ─────── */
#chat{
  flex:1;min-width:0;
  display:flex;flex-direction:column;
  overflow:hidden;
  padding:0;gap:14px;
  max-width:0;          /* collapsed in idle */
  opacity:0;
  pointer-events:none;
  align-self:stretch;
  height:100%;
  min-height:0;
  transition:
    max-width 0.65s cubic-bezier(0.34,1.56,0.64,1),
    opacity   0.4s ease 0.35s,
    padding   0.65s ease;
}
body.session-active #chat{
  max-width:100%;       /* expand — concrete value, fully animatable */
  align-self:stretch;
  height:100%;
  min-height:0;
  max-height:100%;
  overflow-y:auto;
  overflow-x:hidden;
  opacity:1;
  pointer-events:auto;
  padding:24px 28px;
}
#chat::-webkit-scrollbar{width:6px}
#chat::-webkit-scrollbar-track{background:transparent}
#chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
#chat::-webkit-scrollbar-thumb:hover{background:var(--muted)}

/* ── Bubbles ────────────────────────────────────────────────────────── */
.bubble{
  max-width:75%;padding:12px 16px;border-radius:8px;
  font-size:.88rem;line-height:1.6;animation:pop .2s ease;
  word-wrap:break-word;
}
@keyframes pop{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.bubble.kate{background:var(--surface2);border:1px solid var(--border);align-self:flex-start;border-bottom-left-radius:2px}
.bubble.user{background:#0e3639;border:1px solid #144e52;align-self:flex-end;border-bottom-right-radius:2px}
.bname{font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;color:var(--muted);display:flex;align-items:center;gap:5px;}
.bubble.kate .bname{color:var(--accent-light);}
.bubble.user .bname{color:#5eead4;}

/* ── Empty chat state ───────────────────────────────────────────────── */
.empty{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;color:var(--muted);font-size:.85rem;text-align:center;
}
.empty .ico{font-size:2rem;opacity:.5;display:flex;align-items:center;justify-content:center;}

/* ── Footer ─────────────────────────────────────────────────────────── */
footer{
  text-align:center;padding:8px;font-size:.70rem;color:var(--muted);
  border-top:1px solid var(--border);
  background:var(--surface);flex-shrink:0;
}

/* Buttons: hide stop/clear in idle smoothly */
.btn-stop,.btn-clear{
  overflow:hidden;max-height:0;opacity:0;pointer-events:none;margin-top:0;
  transition:max-height 0.7s ease,opacity 0.5s ease,margin-top 0.5s ease;
}
body.session-active .btn-stop,body.session-active .btn-clear{
  max-height:60px;opacity:1;pointer-events:auto;
}
.tips-section{opacity:0.85;}

/* Responsive */
/* Responsive rules for mobile phones */
@media (max-width: 768px) {
  body {
    height: 100vh;
    height: 100dvh;
  }
  header {
    padding: 10px 14px;
    gap: 8px;
    flex-wrap: wrap;
  }
  .brand {
    gap: 8px;
  }
  header h1 {
    font-size: 0.95rem;
  }
  header p {
    font-size: 0.65rem;
  }
  .pill {
    margin-left: auto;
  }
  main {
    flex-direction: column;
    align-items: stretch;
  }
  body.session-active main {
    flex-direction: column;
  }
  #chat {
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 16px;
    height: 55vh;
  }
  aside {
    padding: 14px;
    max-height: 45vh;
    width: 100%;
    transform: none;
  }
  .bubble {
    max-width: 85%;
  }
}
</style>
</head>
<body>

<header>
  <div class="brand">
    <svg class="static-logo" width="34" height="34" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AI Tutor Logo">
      <defs>
        <linearGradient id="hdrGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#14b8a6"/>
          <stop offset="50%" stop-color="#0f766e"/>
          <stop offset="100%" stop-color="#042f2e"/>
        </linearGradient>
        <linearGradient id="hdrSpark" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#5eead4"/>
          <stop offset="100%" stop-color="#14b8a6"/>
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="10" fill="url(#hdrGrad)" stroke="#14b8a6" stroke-width="1.5" stroke-opacity="0.4"/>
      <circle cx="24" cy="24" r="14" stroke="#5eead4" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.6"/>
      <path d="M14 24h3 M19 19v10 M24 15v18 M29 19v10 M34 24h3" stroke="#f1f5f9" stroke-width="2.2" stroke-linecap="round"/>
      <path d="M33 11L34.2 13.8L37 15L34.2 16.2L33 19L31.8 16.2L29 15L31.8 13.8Z" fill="url(#hdrSpark)"/>
    </svg>
    <div class="header-text">
      <h1>AI Tutor</h1>
      <p>Real-time English coaching</p>
    </div>
  </div>

  <div class="pill">
    <div class="dot" id="dot"></div>
    <span id="statusTxt">Disconnected</span>
  </div>
</header>

<main>
  <aside>
    <!-- Shown only in active session -->
    <h2 class="aside-title">Controls</h2>
    <!-- Shown only in idle/centered mode -->
    <div class="idle-hero">
      <div style="margin-bottom:6px;display:flex;align-items:center;justify-content:center;">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="var(--accent-light)" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      </div>
      <h3>AI Tutor</h3>
      <p>Real-time English coaching<br>powered by Gemini Live</p>
    </div>
    <div class="orb-wrap">
      <div class="orb" id="orb">
        <svg class="orb-logo" id="orbLogo" width="100%" height="100%" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="AI Tutor Avatar">
          <defs>
            <radialGradient id="orbGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#144e52" stop-opacity="0.85"/>
              <stop offset="65%" stop-color="#0e2a2e" stop-opacity="0.95"/>
              <stop offset="100%" stop-color="#0b0f17"/>
            </radialGradient>
            <linearGradient id="orbBorderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#2dd4bf"/>
              <stop offset="50%" stop-color="#0f766e"/>
              <stop offset="100%" stop-color="#134e4a"/>
            </linearGradient>
            <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#5eead4"/>
              <stop offset="100%" stop-color="#14b8a6"/>
            </linearGradient>
            <filter id="orbGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
          </defs>
          <circle cx="48" cy="48" r="46" fill="url(#orbGrad)" stroke="url(#orbBorderGrad)" stroke-width="2"/>
          <circle class="logo-orbit" id="logoOrbit" cx="48" cy="48" r="34" stroke="#14b8a6" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.6"/>
          <circle cx="48" cy="48" r="22" stroke="#5eead4" stroke-width="1" opacity="0.35"/>
          <g class="logo-bars" id="logoBars" stroke="#f1f5f9" stroke-width="2.8" stroke-linecap="round">
            <line x1="32" y1="48" x2="32" y2="48"/>
            <line x1="40" y1="42" x2="40" y2="54"/>
            <line x1="48" y1="36" x2="48" y2="60"/>
            <line x1="56" y1="42" x2="56" y2="54"/>
            <line x1="64" y1="48" x2="64" y2="48"/>
          </g>
          <path class="logo-spark" id="logoSpark" d="M66 22L67.8 27.5L73.5 29.2L67.8 31L66 36.5L64.2 31L58.5 29.2L64.2 27.5Z" fill="url(#sparkGrad)" filter="url(#orbGlow)"/>
        </svg>
      </div>
      <span class="kate-label" id="kateState">Idle</span>
      <div class="waveform" id="viz"><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
      <div class="mic-row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> <div class="mic-track"><div class="mic-fill" id="micFill"></div></div> Mic</div>
    </div>
    <div class="btns">
      <button class="btn btn-start" id="btnStart"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"/></svg> Start Session</button>
      <button class="btn btn-stop"  id="btnStop"  disabled><svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true"><rect width="18" height="18" x="3" y="3" rx="2"/></svg> End Session</button>
      <button class="btn btn-clear" id="btnClear" disabled><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg> Clear Chat</button>
    </div>
    <div class="tips-section">
      <h2 style="margin-bottom:8px">Tips</h2>
      <div class="tips">
        <p>&bull; <strong>Use headphones</strong> to prevent echo.</p>
        <p>&bull; Speak clearly at a natural pace.</p>
        <p>&bull; Ask your tutor to roleplay interviews or demos.</p>
        <p>&bull; Grammar &amp; technical errors corrected live.</p>
      </div>
    </div>
  </aside>
  <section id="chat">
    <div class="empty" id="empty">
      <div class="ico"><svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg></div>
      <p><strong>Click "Start Session"</strong> to begin.</p>
      <p>Your AI tutor will introduce herself and guide you.</p>
    </div>
  </section>
</main>

<footer>Gemini 3.1 Flash Live &middot; &copy; 2026</footer>

<script>
const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
const MIC_RATE = 16000;
const OUT_RATE = 24000;

const btnStart  = document.getElementById('btnStart');
const btnStop   = document.getElementById('btnStop');
const btnClear  = document.getElementById('btnClear');
const chatEl    = document.getElementById('chat');
const emptyEl   = document.getElementById('empty');
const dot       = document.getElementById('dot');
const statusTxt = document.getElementById('statusTxt');
const orb       = document.getElementById('orb');
const kateState = document.getElementById('kateState');
const micFill   = document.getElementById('micFill');
const vizEl     = document.getElementById('viz');

let ws=null, audioCtx=null, micStream=null, processor=null, analyser=null;
let kateSpeaking=false, rafId=null, activeAudioChunks=0, nextPlayTime=0, pendingKateTimeouts=[];
let userScrolledUp = false;

// ── Motion.dev Animations ────────────────────────────────────────────────
let motionAnimate = null;

async function initMotion() {
  try {
    const motion = await import('https://cdn.jsdelivr.net/npm/motion@latest/+esm');
    motionAnimate = motion.animate;
  } catch (err) {
    console.warn('Motion.dev CDN unavailable, using Web Animations API fallback', err);
  }
  animateLogoEntrance();
  startIdleMotion();
}

function animateLogoEntrance() {
  const orbEl = document.getElementById('orbLogo');
  if (!orbEl) return;
  if (motionAnimate) {
    motionAnimate(orbEl, { scale: [0.85, 1], opacity: [0, 1] }, { duration: 0.8, ease: 'easeOut' });
  } else {
    orbEl.animate([{ transform: 'scale(0.85)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: 800, easing: 'ease-out', fill: 'forwards' });
  }
}

let idleMotionStop = null;
function startIdleMotion() {
  const orbit = document.getElementById('logoOrbit');
  const spark = document.getElementById('logoSpark');
  if (motionAnimate) {
    if (orbit) {
      idleMotionStop = motionAnimate(orbit, { rotate: [0, 360] }, { duration: 24, repeat: Infinity, ease: 'linear' });
    }
    if (spark) {
      motionAnimate(spark, { scale: [1, 1.25, 1], opacity: [0.75, 1, 0.75] }, { duration: 3, repeat: Infinity, ease: 'easeInOut' });
    }
  } else {
    if (orbit) {
      orbit.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }], { duration: 24000, iterations: Infinity });
    }
    if (spark) {
      spark.animate([{ transform: 'scale(1)', opacity: 0.75 }, { transform: 'scale(1.25)', opacity: 1 }, { transform: 'scale(1)', opacity: 0.75 }], { duration: 3000, iterations: Infinity, easing: 'ease-in-out' });
    }
  }
}

let speakingMotionControls = null;
function setSpeakingMotion(speaking) {
  const orbEl = document.getElementById('orbLogo');
  const bars = document.getElementById('logoBars');
  if (!orbEl) return;

  if (speaking) {
    if (motionAnimate) {
      if (!speakingMotionControls) {
        speakingMotionControls = motionAnimate(orbEl, { scale: [1, 1.06, 0.98, 1.04, 1] }, { duration: 1.3, repeat: Infinity, ease: 'easeInOut' });
      }
      if (bars) {
        motionAnimate(bars, { opacity: [0.7, 1, 0.7] }, { duration: 0.5, repeat: Infinity, ease: 'easeInOut' });
      }
    } else {
      orbEl.classList.add('speaking');
    }
  } else {
    if (speakingMotionControls) {
      speakingMotionControls.stop();
      speakingMotionControls = null;
    }
    if (motionAnimate) {
      motionAnimate(orbEl, { scale: 1 }, { duration: 0.35, ease: 'easeOut' });
      if (bars) {
        motionAnimate(bars, { opacity: 1 }, { duration: 0.35, ease: 'easeOut' });
      }
    } else {
      orbEl.classList.remove('speaking');
    }
  }
}

initMotion();

// ── Auto-scroll tracking ─────────────────────────────────────────────────
chatEl.addEventListener('scroll', () => {
  const distance = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight;
  userScrolledUp = distance > 80;
});

function scrollToBottom(force = false) {
  if (!chatEl) return;
  if (!userScrolledUp || force) {
    requestAnimationFrame(() => {
      chatEl.scrollTop = chatEl.scrollHeight;
    });
  }
}

function setStatus(s){
  dot.className='dot '+s;
  statusTxt.textContent={connecting:'Connecting…',live:'Live',disconnected:'Disconnected',error:'Error'}[s]??s;
}

function removeEmpty(){ if(emptyEl&&emptyEl.parentNode) emptyEl.remove(); }
function escHtml(t){ const d=document.createElement('div');d.textContent=t;return d.innerHTML; }
function scheduleKateBubble(text){
  if(!audioCtx){ addBubble('kate', text); return; }
  const delayMs = Math.max(0, (nextPlayTime - audioCtx.currentTime) * 1000);
  if(delayMs < 30){
    addBubble('kate', text);
  } else {
    const id = setTimeout(()=>addBubble('kate', text), delayMs);
    pendingKateTimeouts.push(id);
  }
}

function clearPendingKateBubbles(){
  pendingKateTimeouts.forEach(id=>clearTimeout(id));
  pendingKateTimeouts = [];
}

function addBubble(who,text){
  removeEmpty();
  const last=chatEl.lastElementChild;
  if(last&&last.classList.contains(who)&&last.classList.contains('bubble')){
    const span = last.querySelector('span');
    if (span) span.textContent += ' ' + text;
  } else {
    const b=document.createElement('div');
    b.className=`bubble ${who}`;
    b.innerHTML=`<div class="bname">${who==='kate'?'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="18" height="18" x="3" y="2" rx="2"/><line x1="8" x2="8" y1="2" y2="4"/><line x1="16" x2="16" y1="2" y2="4"/><circle cx="9" cy="11" r="1" fill="currentColor"/><circle cx="15" cy="11" r="1" fill="currentColor"/><path d="M9 16h6"/></svg> Tutor':'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg> You'}</div><span>${escHtml(text)}</span>`;
    chatEl.appendChild(b);
  }
  if (who === 'user') userScrolledUp = false;
  scrollToBottom(who === 'user');
}

function drawViz(){
  rafId=requestAnimationFrame(drawViz);
  if(!analyser||!vizEl) return;
  const bars=vizEl.querySelectorAll('.bar');
  const d=new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(d);
  const total=bars.length;
  bars.forEach((bar,i)=>{
    const v=d[Math.floor(i*d.length/total)]/255;
    const h=Math.max(4, Math.round(v*44));
    bar.style.height=h+'px';
    const alpha=0.35+v*0.65;
    bar.style.opacity=alpha;
  });
}

function playPCM(buf){
  if(!audioCtx) return;
  const i16=new Int16Array(buf);
  const f32=new Float32Array(i16.length);
  for(let i=0;i<i16.length;i++) f32[i]=i16[i]/32768;
  const ab=audioCtx.createBuffer(1,f32.length,OUT_RATE);
  ab.copyToChannel(f32,0);
  const src=audioCtx.createBufferSource();
  src.buffer=ab; src.connect(audioCtx.destination);
  const startTime = Math.max(audioCtx.currentTime, nextPlayTime);
  src.start(startTime);
  nextPlayTime = startTime + ab.duration;
  activeAudioChunks++;
  kateSpeaking=true; orb.classList.add('speaking'); vizEl&&vizEl.classList.add('speaking'); kateState.textContent='Speaking…';
  setSpeakingMotion(true);
  src.onended=()=>{
    activeAudioChunks--;
    if (activeAudioChunks <= 0) {
      activeAudioChunks = 0;
      setTimeout(() => {
        if (activeAudioChunks <= 0) {
          kateSpeaking=false; orb.classList.remove('speaking'); vizEl&&vizEl.classList.remove('speaking'); kateState.textContent='Listening';
          setSpeakingMotion(false);
        }
      }, 600);
    }
  };
}

async function startMic(){
  micStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});
  const source=audioCtx.createMediaStreamSource(micStream);
  analyser=audioCtx.createAnalyser(); analyser.fftSize=256;
  source.connect(analyser);
  const bufSz=4096;
  processor=audioCtx.createScriptProcessor(bufSz,1,1);
  source.connect(processor); processor.connect(audioCtx.destination);
  processor.onaudioprocess=e=>{
    if(!ws||ws.readyState!==WebSocket.OPEN||kateSpeaking) return;
    const f32=e.inputBuffer.getChannelData(0);
    let sum=0; for(let i=0;i<f32.length;i++) sum+=f32[i]*f32[i];
    micFill.style.width=Math.min(100,Math.sqrt(sum/f32.length)*900)+'%';
    const ratio=MIC_RATE/audioCtx.sampleRate;
    const out=new Int16Array(Math.round(f32.length*ratio));
    for(let i=0;i<out.length;i++){
      const si=Math.min(Math.round(i/ratio),f32.length-1);
      out[i]=Math.max(-32768,Math.min(32767,f32[si]*32767));
    }
    ws.send(out.buffer);
  };
}

function stopMic(){
  if(processor){processor.disconnect();processor=null;}
  if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null;}
  analyser=null; micFill.style.width='0%';
}

async function connect(){
  setStatus('connecting'); btnStart.disabled=true;
  audioCtx=new AudioContext({sampleRate:OUT_RATE});
  ws=new WebSocket(WS_URL); ws.binaryType='arraybuffer';

  ws.onopen=async()=>{
    setStatus('live');
    document.body.classList.add('session-active');
    btnStop.disabled=false; btnClear.disabled=false;
    kateState.textContent='Connecting…';
    nextPlayTime = audioCtx.currentTime;
    userScrolledUp = false;
    scrollToBottom(true);
    try{ await startMic(); }
    catch(e){ console.error('Mic error:',e); addBubble('kate','Mic error: '+e.name+' — '+e.message); }
    drawViz();
  };

  ws.onmessage=e=>{
    if(e.data instanceof ArrayBuffer){ playPCM(e.data); return; }
    try{
      const m=JSON.parse(e.data);
      if(m.type==='kate_transcript'&&m.text.trim()) scheduleKateBubble(m.text);
      if(m.type==='user_transcript'&&m.text.trim()) addBubble('user',m.text);
      if(m.type==='error') addBubble('kate','Error: '+m.message);
      if(m.type==='interrupted'){
        activeAudioChunks=0; kateSpeaking=false; orb.classList.remove('speaking');
        vizEl&&vizEl.classList.remove('speaking'); kateState.textContent='Listening';
        nextPlayTime=audioCtx.currentTime; clearPendingKateBubbles();
        setSpeakingMotion(false);
      }
    }catch{}
  };

  ws.onclose=()=>{
    setStatus('disconnected'); stopMic();
    cancelAnimationFrame(rafId); rafId=null;
    clearPendingKateBubbles();
    document.body.classList.remove('session-active');
    orb.classList.remove('speaking'); kateState.textContent='Idle';
    setSpeakingMotion(false);
    btnStart.disabled=false; btnStop.disabled=true;
  };

  ws.onerror=()=>{ setStatus('error'); addBubble('kate','Connection error. Is the server running on port 8000?'); };
}

function disconnect(){ ws&&ws.close(); audioCtx&&(audioCtx.close(),audioCtx=null); }

btnStart.addEventListener('click',connect);
btnStop.addEventListener('click',disconnect);
btnClear.addEventListener('click',()=>{
  chatEl.innerHTML='<div class="empty"><div class="ico"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div><p>Chat cleared — keep talking!</p></div>';
  userScrolledUp = false;
  scrollToBottom(true);
});
</script>
</body>
</html>"""


# FastAPI App
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="English Communication Tutor")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the embedded UI."""
    return HTMLResponse(content=HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


# ────────────────────────────────────────────────────────────────────────────
# WebSocket — browser ↔ Gemini Live bridge
# ────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] client connected")

    client = genai.Client(api_key=API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
        output_audio_transcription={},
        input_audio_transcription={},
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("[ws] Gemini Live session opened")

            # Silent pulse → triggers Kate's self-introduction immediately
            await session.send_realtime_input(
                audio=types.Blob(data=b"\x00" * 320, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )

            async def mic_to_gemini():
                while True:
                    data = await websocket.receive_bytes()
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
                    )

            async def gemini_to_browser():
                while True:
                    async for response in session.receive():
                        if response.data:
                            await websocket.send_bytes(response.data)

                        if response.server_content:
                            sc = response.server_content
                            if sc.input_transcription and sc.input_transcription.text:
                                await websocket.send_text(json.dumps({
                                    "type": "user_transcript",
                                    "text": sc.input_transcription.text
                                }))
                            if sc.output_transcription and sc.output_transcription.text:
                                await websocket.send_text(json.dumps({
                                    "type": "kate_transcript",
                                    "text": sc.output_transcription.text
                                }))
                            if getattr(sc, "interrupted", False):
                                await websocket.send_text(json.dumps({"type": "interrupted"}))

            await asyncio.gather(mic_to_gemini(), gemini_to_browser())

    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except Exception as exc:
        print(f"[ws] error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)