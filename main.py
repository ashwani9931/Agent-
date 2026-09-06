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
<title>Kate – English Tutor</title>
<style>
:root{
  --bg:#000;--surface:#0a0a0a;--surface2:#141414;
  --border:#2a2a2a;--text:#f0f0f0;--muted:#666;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}

header{
  display:flex;align-items:center;gap:14px;
  padding:14px 24px;
  background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0;
}
header h1{font-size:1.1rem;font-weight:700;letter-spacing:.02em}
header p{font-size:.73rem;color:var(--muted);margin-top:2px}
.pill{
  margin-left:auto;padding:5px 14px;border-radius:20px;
  font-size:.73rem;font-weight:600;
  background:var(--surface2);border:1px solid var(--border);
  display:flex;align-items:center;gap:6px;
}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:background .3s}
.dot.live{background:#fff;box-shadow:0 0 6px rgba(255,255,255,.5)}
.dot.connecting{background:#aaa;animation:blink 1s infinite}
.dot.error{background:#888}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

main{flex:1;display:grid;grid-template-columns:1fr 300px;overflow:hidden}

#chat{
  display:flex;flex-direction:column;overflow-y:auto;
  padding:20px 24px;gap:14px;scroll-behavior:smooth;
  border-right:1px solid var(--border);
}
#chat::-webkit-scrollbar{width:5px}
#chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

.bubble{
  max-width:75%;padding:11px 15px;border-radius:16px;
  font-size:.88rem;line-height:1.6;animation:pop .2s ease;
}
@keyframes pop{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.bubble.kate{background:var(--surface2);border:1px solid var(--border);align-self:flex-start;border-bottom-left-radius:3px}
.bubble.user{background:#111;border:1px solid #2a2a2a;align-self:flex-end;border-bottom-right-radius:3px}
.bname{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;color:var(--muted)}

.empty{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:8px;color:var(--muted);font-size:.85rem;text-align:center;
}

aside{
  background:var(--surface);
  display:flex;flex-direction:column;padding:20px 16px;gap:18px;overflow-y:auto;
}
aside h2{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}

.orb-wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.orb{
  width:64px;height:64px;border-radius:50%;
  background:#111;border:2px solid #2a2a2a;
  transition:border-color .3s,box-shadow .3s;
}
.orb.speaking{border-color:#fff;animation:orbpulse 1.1s ease-in-out infinite}
@keyframes orbpulse{
  0%,100%{box-shadow:0 0 0 4px rgba(255,255,255,.1),0 0 0 10px rgba(255,255,255,.04)}
  50%{box-shadow:0 0 0 8px rgba(255,255,255,.15),0 0 0 18px rgba(255,255,255,.05)}
}
.kate-label{font-size:.75rem;color:var(--muted);font-weight:500}

canvas#viz{
  width:100%;height:64px;border-radius:8px;
  background:var(--surface2);border:1px solid var(--border);display:block;
}

.mic-row{display:flex;align-items:center;gap:8px;font-size:.72rem;color:var(--muted)}
.mic-track{flex:1;height:4px;border-radius:2px;background:var(--border);overflow:hidden}
.mic-fill{height:100%;width:0%;border-radius:2px;background:#fff;transition:width .05s}

.btns{display:flex;flex-direction:column;gap:8px}
.btn{
  width:100%;padding:10px;border:none;border-radius:8px;
  font-size:.83rem;font-weight:600;cursor:pointer;
  transition:all .15s;display:flex;align-items:center;justify-content:center;gap:7px;
}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-start{background:#fff;color:#000}
.btn-start:hover:not(:disabled){background:#d0d0d0}
.btn-stop{background:transparent;color:#888;border:1px solid #2a2a2a}
.btn-stop:hover:not(:disabled){background:#111;color:#ccc;border-color:#555}
.btn-clear{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-clear:hover:not(:disabled){background:var(--surface2);color:var(--text)}

</style>
</head>
<body>

<header>
  <div class="pill">
    <div class="dot" id="dot"></div>
    <span id="statusTxt">Disconnected</span>
  </div>
</header>

<main>
  <section id="chat">
    <div class="empty" id="empty">
      <p><strong>Click "Start Session"</strong> to begin.</p>
      <p>Kate will introduce herself and guide you.</p>
    </div>
  </section>

  <aside>
    <h2>Controls</h2>
    <div class="orb-wrap">
      <div class="orb" id="orb"></div>
      <span class="kate-label" id="kateState">Idle</span>
      <canvas id="viz"></canvas>
      <div class="mic-row">Mic <div class="mic-track"><div class="mic-fill" id="micFill"></div></div></div>
    </div>
    <div class="btns">
      <button class="btn btn-start" id="btnStart">Start Session</button>
      <button class="btn btn-stop"  id="btnStop"  disabled>End Session</button>
      <button class="btn btn-clear" id="btnClear" disabled>Clear Chat</button>
    </div>
  </aside>
</main>

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
const canvas    = document.getElementById('viz');
const ctx2d     = canvas.getContext('2d');

let ws=null, audioCtx=null, micStream=null, processor=null, analyser=null;
let kateSpeaking=false, rafId=null;

// ── Playback scheduling state ────────────────────────────────────────────
// Chunks from the model arrive faster than they play. Instead of firing
// each one with src.start() (which plays it *immediately*, overlapping
// with whatever is still playing — this was the "repeating/garbled audio"
// bug), we schedule each buffer to start exactly when the previous one
// ends, using the AudioContext's own clock.
let nextStartTime = 0;
let scheduledSources = [];
let activeSources = 0;
let speakingClearTimer = null;
const SPEAKING_SILENCE_DEBOUNCE_MS = 300; // mirrors the debounce in app.py

function setStatus(s){
  dot.className='dot '+s;
  statusTxt.textContent={connecting:'Connecting…',live:'Live',disconnected:'Disconnected',error:'Error'}[s]??s;
}

function setKateSpeaking(v){
  kateSpeaking = v;
  orb.classList.toggle('speaking', v);
  kateState.textContent = v ? 'Speaking…' : 'Listening';
}

function removeEmpty(){ if(emptyEl&&emptyEl.parentNode) emptyEl.remove(); }
function escHtml(t){ const d=document.createElement('div');d.textContent=t;return d.innerHTML; }
function addBubble(who,text){
  removeEmpty();
  const last=chatEl.lastElementChild;
  if(last&&last.classList.contains(who)&&last.classList.contains('bubble')){
    last.querySelector('span').textContent+=' '+text;
  } else {
    const b=document.createElement('div');
    b.className=`bubble ${who}`;
    b.innerHTML=`<div class="bname">${who==='kate'?'Kate':'You'}</div><span>${escHtml(text)}</span>`;
    chatEl.appendChild(b);
  }
  chatEl.scrollTop=chatEl.scrollHeight;
}

function resizeCanvas(){
  canvas.width=canvas.offsetWidth*devicePixelRatio;
  canvas.height=canvas.offsetHeight*devicePixelRatio;
}
resizeCanvas(); window.addEventListener('resize',resizeCanvas);

function drawViz(){
  rafId=requestAnimationFrame(drawViz);
  const W=canvas.width,H=canvas.height;
  ctx2d.clearRect(0,0,W,H);
  if(!analyser) return;
  const d=new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(d);
  const bars=44, bw=W/bars;
  for(let i=0;i<bars;i++){
    const v=d[Math.floor(i*d.length/bars)]/255;
    const h=v*H*.88;
    const br=Math.round(40+v*200);
    ctx2d.fillStyle=`rgb(${br},${br},${br})`;
    ctx2d.beginPath();
    if(ctx2d.roundRect) ctx2d.roundRect(i*bw+2,H-h,bw-4,h,2);
    else ctx2d.rect(i*bw+2,H-h,bw-4,h);
    ctx2d.fill();
  }
}

function resetPlaybackClock(){
  nextStartTime = audioCtx ? audioCtx.currentTime : 0;
}

// How far "behind" the audio playback currently is, in ms. Text transcripts
// arrive over the WebSocket as soon as Gemini produces them, but the matching
// audio is queued to play sequentially (see playPCM) and may not start for a
// little while if there's a backlog. Delaying the caption by this amount
// keeps what's on screen in sync with what you actually hear.
function audioBacklogMs(){
  if(!audioCtx) return 0;
  return Math.max(0, (nextStartTime - audioCtx.currentTime) * 1000);
}

function stopAllPlayback(){
  // Used on a genuine model interruption (server_content.interrupted) —
  // stop everything queued so Kate doesn't keep talking over the user.
  scheduledSources.forEach(s => { try { s.onended = null; s.stop(); } catch (_) {} });
  scheduledSources = [];
  activeSources = 0;
  if (speakingClearTimer) { clearTimeout(speakingClearTimer); speakingClearTimer = null; }
  resetPlaybackClock();
  setKateSpeaking(false);
}

function playPCM(buf){
  if(!audioCtx) return;
  const i16=new Int16Array(buf);
  const f32=new Float32Array(i16.length);
  for(let i=0;i<i16.length;i++) f32[i]=i16[i]/32768;
  const ab=audioCtx.createBuffer(1,f32.length,OUT_RATE);
  ab.copyToChannel(f32,0);

  const src=audioCtx.createBufferSource();
  src.buffer=ab;
  src.connect(audioCtx.destination);

  // Schedule back-to-back instead of playing immediately.
  const now = audioCtx.currentTime;
  if (nextStartTime < now) nextStartTime = now;
  src.start(nextStartTime);
  nextStartTime += ab.duration;

  scheduledSources.push(src);
  activeSources++;
  if (speakingClearTimer) { clearTimeout(speakingClearTimer); speakingClearTimer = null; }
  setKateSpeaking(true);

  src.onended = () => {
    activeSources = Math.max(0, activeSources - 1);
    scheduledSources = scheduledSources.filter(s => s !== src);
    if (activeSources === 0) {
      // Debounce: a new chunk may arrive milliseconds later (network jitter),
      // don't flip the mic back on for a brief real gap between chunks.
      if (speakingClearTimer) clearTimeout(speakingClearTimer);
      speakingClearTimer = setTimeout(() => {
        if (activeSources === 0) setKateSpeaking(false);
      }, SPEAKING_SILENCE_DEBOUNCE_MS);
    }
  };
}

async function startMic(){
  micStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});
  const source=audioCtx.createMediaStreamSource(micStream);
  analyser=audioCtx.createAnalyser(); analyser.fftSize=256;
  source.connect(analyser);
  // createScriptProcessor only accepts: 256,512,1024,2048,4096,8192,16384
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
  resetPlaybackClock();
  ws=new WebSocket(WS_URL); ws.binaryType='arraybuffer';

  ws.onopen=async()=>{
    setStatus('live');
    btnStop.disabled=false; btnClear.disabled=false;
    kateState.textContent='Connecting…';
    try{ await startMic(); }
    catch(e){ console.error('Mic error:',e); addBubble('kate','Mic error: '+e.name+' — '+e.message); }
    drawViz();
  };

  ws.onmessage=e=>{
    if(e.data instanceof ArrayBuffer){ playPCM(e.data); return; }
    try{
      const m=JSON.parse(e.data);
      if(m.type==='kate_transcript'&&m.text.trim()){
        // Delay by the current audio backlog so the caption lands in sync
        // with the audio, instead of appearing the instant text arrives.
        setTimeout(()=>addBubble('kate',m.text), audioBacklogMs());
      }
      if(m.type==='user_transcript'&&m.text.trim()) addBubble('user',m.text);
      if(m.type==='error') addBubble('kate','Error: '+m.message);
      if(m.type==='interrupted') stopAllPlayback();
    }catch{}
  };

  ws.onclose=()=>{
    setStatus('disconnected'); stopMic(); stopAllPlayback();
    cancelAnimationFrame(rafId); rafId=null;
    kateState.textContent='Idle';
    btnStart.disabled=false; btnStop.disabled=true;
  };

  ws.onerror=()=>{ setStatus('error'); addBubble('kate','Connection error. Is the server running on port 8000?'); };
}

function disconnect(){
  ws&&ws.close();
  stopAllPlayback();
  audioCtx&&(audioCtx.close(),audioCtx=null);
}

btnStart.addEventListener('click',connect);
btnStop.addEventListener('click',disconnect);
btnClear.addEventListener('click',()=>{
  chatEl.innerHTML='<div class="empty"><p>Chat cleared — keep talking!</p></div>';
});
</script>
</body>
</html>"""


# FastAPI App
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Kate – English Communication Tutor")


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