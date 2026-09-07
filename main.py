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
<title>English Communication Tutor</title>
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
.logo-img {
  height: 48px;
  width: auto;
  object-fit: contain;
}
.header-text {
  display: flex;
  flex-direction: column;
}
header h1{font-size:1.1rem;font-weight:700;letter-spacing:.02em;margin:0;}
header p{font-size:.73rem;color:var(--muted);margin-top:2px;}
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
  word-wrap: break-word;
}
@keyframes pop{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.bubble.kate{background:var(--surface2);border:1px solid var(--border);align-self:flex-start;border-bottom-left-radius:3px}
.bubble.user{background:#111;border:1px solid #2a2a2a;align-self:flex-end;border-bottom-right-radius:3px}
.bname{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;color:var(--muted)}

.empty{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;color:var(--muted);font-size:.85rem;text-align:center;
}
.empty .ico{font-size:2.4rem;opacity:.4}

aside{
  background:var(--surface);
  display:flex;flex-direction:column;padding:20px 16px;gap:18px;overflow-y:auto;
}
aside h2{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}

.orb-wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
.orb{
  width:76px;height:76px;border-radius:50%;
  background:transparent;border:none;
  display:flex;align-items:center;justify-content:center;font-size:1.9rem;
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

.tips{font-size:.72rem;color:var(--muted);line-height:1.7}
.tips strong{color:#aaa}

footer{
  text-align:center;padding:7px;font-size:.67rem;color:#333;
  border-top:1px solid var(--border);background:var(--surface);flex-shrink:0;
}

/* Responsive rules for mobile phones */
@media (max-width: 768px) {
  body {
    height: 100vh;
    height: 100dvh; /* Better mobile handling */
  }
  header {
    padding: 10px 14px;
    gap: 8px;
    flex-wrap: wrap;
  }
  .logo-img {
    height: 36px;
  }
  header h1 {
    font-size: 0.95rem;
  }
  header p {
    font-size: 0.65rem;
  }
  .pill {
    margin-left: 0;
    margin-top: 4px;
    width: 100%;
    justify-content: center;
  }
  main {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr auto;
  }
  #chat {
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 16px;
  }
  aside {
    padding: 14px;
    max-height: 45vh;
  }
  .bubble {
    max-width: 85%;
  }
}
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
      <div class="ico">&#x1F399;</div>
      <p><strong>Click "Start Session"</strong> to begin.</p>
      <p>Your AI tutor will introduce herself and guide you.</p>
    </div>
  </section>

  <aside>
    <h2>Controls</h2>
    <div class="orb-wrap">
      <div class="orb" id="orb"><img src="data:image/png;base64,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" alt="Tutor" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" /></div>
      <span class="kate-label" id="kateState">Idle</span>
      <canvas id="viz"></canvas>
      <div class="mic-row">&#x1F3A4; <div class="mic-track"><div class="mic-fill" id="micFill"></div></div> Mic</div>
    </div>
    <div class="btns">
      <button class="btn btn-start" id="btnStart">&#x25B6; Start Session</button>
      <button class="btn btn-stop"  id="btnStop"  disabled>&#x23F9; End Session</button>
      <button class="btn btn-clear" id="btnClear" disabled>&#x1F5D1; Clear Chat</button>
    </div>
    <div>
      <h2 style="margin-bottom:8px">Tips</h2>
      <div class="tips">
        <p>&bull; <strong>Use headphones</strong> to prevent echo.</p>
        <p>&bull; Speak clearly at a natural pace.</p>
        <p>&bull; Ask your tutor to roleplay interviews or demos.</p>
        <p>&bull; Grammar &amp; technical errors corrected live.</p>
      </div>
    </div>
  </aside>
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
const canvas    = document.getElementById('viz');
const ctx2d     = canvas.getContext('2d');

let ws=null, audioCtx=null, micStream=null, processor=null, analyser=null;
let kateSpeaking=false, rafId=null;

function setStatus(s){
  dot.className='dot '+s;
  statusTxt.textContent={connecting:'Connecting…',live:'Live',disconnected:'Disconnected',error:'Error'}[s]??s;
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
    b.innerHTML=`<div class="bname">${who==='kate'?'Tutor':'You'}</div><span>${escHtml(text)}</span>`;
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

function playPCM(buf){
  if(!audioCtx) return;
  const i16=new Int16Array(buf);
  const f32=new Float32Array(i16.length);
  for(let i=0;i<i16.length;i++) f32[i]=i16[i]/32768;
  const ab=audioCtx.createBuffer(1,f32.length,OUT_RATE);
  ab.copyToChannel(f32,0);
  const src=audioCtx.createBufferSource();
  src.buffer=ab; src.connect(audioCtx.destination); src.start();
  kateSpeaking=true; orb.classList.add('speaking'); kateState.textContent='Speaking…';
  src.onended=()=>{ kateSpeaking=false; orb.classList.remove('speaking'); kateState.textContent='Listening'; };
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
      if(m.type==='kate_transcript'&&m.text.trim()) addBubble('kate',m.text);
      if(m.type==='user_transcript'&&m.text.trim()) addBubble('user',m.text);
      if(m.type==='error') addBubble('kate','Error: '+m.message);
      if(m.type==='interrupted'){ kateSpeaking=false; orb.classList.remove('speaking'); kateState.textContent='Listening'; }
    }catch{}
  };

  ws.onclose=()=>{
    setStatus('disconnected'); stopMic();
    cancelAnimationFrame(rafId); rafId=null;
    orb.classList.remove('speaking'); kateState.textContent='Idle';
    btnStart.disabled=false; btnStop.disabled=true;
  };

  ws.onerror=()=>{ setStatus('error'); addBubble('kate','Connection error. Is the server running on port 8000?'); };
}

function disconnect(){ ws&&ws.close(); audioCtx&&(audioCtx.close(),audioCtx=null); }

btnStart.addEventListener('click',connect);
btnStop.addEventListener('click',disconnect);
btnClear.addEventListener('click',()=>{
  chatEl.innerHTML='<div class="empty"><div class="ico" style="opacity:.4">&#x1F4AC;</div><p>Chat cleared — keep talking!</p></div>';
});
</script>
</body>
</html>"""


# FastAPI App
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="English Communication Tutor")


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