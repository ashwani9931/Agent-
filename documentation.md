# AI Tutor — Technical Documentation

This document describes the internal pipeline of the **Real-time English
Communication Tutor** — how audio flows from your microphone, through the
application, to Google's **Gemini Live API**, and back to your speaker as spoken
feedback from the AI consultant **"Kate."**

It covers both deployment modes:

- **Web app** (`main.py`) — browser ⇄ FastAPI WebSocket ⇄ Gemini Live
- **Terminal app** (`app.py`) — PyAudio ⇄ asyncio tasks ⇄ Gemini Live

---

## 1. High-Level Pipeline

```
┌────────────┐   mic PCM    ┌─────────────┐  realtime   ┌──────────────┐
│            │  16 kHz 16bit│             │  audio in   │              │
│   USER     │─────────────►│ APPLICATION │────────────►│ GEMINI LIVE  │
│ (mic/spkr) │              │  (bridge)   │             │    MODEL     │
│            │◄─────────────│             │◄────────────│   (Kate)     │
└────────────┘   spkr PCM   └─────────────┘  audio +    └──────────────┘
                 24 kHz 16bit              transcripts
```

The application is a thin, bidirectional **bridge**. It does no speech
processing itself — all speech recognition, reasoning, and speech synthesis
happen inside the Gemini Live model. The app's job is to:

1. Capture microphone audio and forward it as 16 kHz PCM.
2. Receive model audio (24 kHz PCM) and play it back.
3. Receive and display transcripts and interruption signals.

---

## 2. Core Concepts

### 2.1 Audio formats

| Direction        | Sample rate | Format         | Channels |
|------------------|-------------|----------------|----------|
| Mic → model      | 16,000 Hz   | 16-bit PCM     | 1 (mono) |
| Model → speaker  | 24,000 Hz   | 16-bit PCM     | 1 (mono) |

These are defined as `SEND_SAMPLE_RATE` and `RECV_SAMPLE_RATE` /
`RECEIVE_SAMPLE_RATE` in both entry points.

### 2.2 The system prompt (Kate's behavior)

A shared `SYSTEM_PROMPT` configures the model persona "Kate." Key rules baked
into the prompt:

- **Speak first** — introduce herself immediately on connect.
- **Listen fully** — let the learner finish before responding.
- **Live feedback** — full-sentence grammar reformulations, technical
  corrections, and pronunciation modeling.
- **Progress the dialogue** — end turns with a follow-up question.

### 2.3 The "speak first" trigger

The model only responds to input. To make Kate introduce herself without the
user speaking first, the server sends a **silent audio pulse** right after the
session opens:

```python
await session.send_realtime_input(
    audio=types.Blob(data=b"\x00" * 320, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
)
```

This nudges the model to take the first turn.

---

## 3. Web App Pipeline (`main.py`)

### 3.1 Components

```
┌───────────────────────── BROWSER (embedded single-page UI) ─────────────────────────┐
│                                                                                      │
│  getUserMedia → AudioContext → ScriptProcessor → resample to 16 kHz → ws.send(PCM)   │
│                                                                                      │
│  ws.onmessage(binary PCM) → playPCM() → AudioContext scheduled playback (24 kHz)     │
│  ws.onmessage(JSON)       → chat bubbles / status / interruption handling            │
│                                                                                      │
└───────────────────────────────────────────┬──────────────────────────────────────────┘
                                             │  WebSocket  /ws
                                             ▼
┌───────────────────────────────── FASTAPI SERVER (main.py) ──────────────────────────┐
│                                                                                      │
│  GET  /         → serves embedded HTML UI                                            │
│  GET  /health   → { status, model }                                                  │
│  WS   /ws       → bridge:                                                             │
│                     ├─ mic_to_gemini()     : browser PCM  → session.send_realtime    │
│                     └─ gemini_to_browser() : session audio → ws binary               │
│                                              session text  → ws JSON (transcripts)   │
│                                                                                      │
└───────────────────────────────────────────┬──────────────────────────────────────────┘
                                             │  genai live session
                                             ▼
                                   ┌────────────────────┐
                                   │  GEMINI LIVE MODEL  │
                                   │  (Kate persona)     │
                                   └────────────────────┘
```

### 3.2 Step-by-step flow

**Connect**

1. User clicks **Start Session** → browser opens `WebSocket /ws`.
2. Server `accept()`s the socket and opens a Gemini Live session with
   `response_modalities=["AUDIO"]` plus input/output transcription enabled.
3. Server sends the silent pulse → Kate's intro begins streaming back.
4. Browser calls `getUserMedia` and starts capturing mic audio.

**Uplink (browser → model)**

```
mic samples (device rate)
   → ScriptProcessor.onaudioprocess
   → compute RMS for mic level meter
   → resample to 16 kHz Int16
   → ws.send(ArrayBuffer)
   → server mic_to_gemini(): websocket.receive_bytes()
   → session.send_realtime_input(Blob pcm;rate=16000)
```

The browser **suppresses uplink while Kate is speaking** (`kateSpeaking`
guard) — a half-duplex strategy that prevents the model from hearing its own
voice through the speakers.

**Downlink (model → browser)**

```
session.receive() yields responses
   ├─ response.data (audio) → ws.send_bytes(PCM 24 kHz)
   └─ response.server_content:
        ├─ input_transcription  → ws JSON { type: "user_transcript" }
        ├─ output_transcription → ws JSON { type: "kate_transcript" }
        └─ interrupted          → ws JSON { type: "interrupted" }
```

**Playback & UI**

- Binary frames → `playPCM()` converts Int16 → Float32 and schedules buffers
  on a shared `nextPlayTime` clock for gapless playback.
- `kate_transcript` bubbles are time-aligned to playback via
  `scheduleKateBubble()` so text appears roughly when it's spoken.
- `user_transcript` renders instantly as a "You" bubble.
- `interrupted` clears queued audio and pending bubbles, resets the playback
  clock, and returns the orb to the listening state.

### 3.3 WebSocket message protocol

| Frame type | Direction        | Payload |
|------------|------------------|---------|
| Binary     | browser → server | 16 kHz Int16 PCM mic chunk |
| Binary     | server → browser | 24 kHz Int16 PCM model audio |
| JSON       | server → browser | `{ "type": "user_transcript", "text": ... }` |
| JSON       | server → browser | `{ "type": "kate_transcript", "text": ... }` |
| JSON       | server → browser | `{ "type": "interrupted" }` |
| JSON       | server → browser | `{ "type": "error", "message": ... }` |

### 3.4 Concurrency

The `/ws` handler runs two coroutines concurrently with `asyncio.gather`:

- `mic_to_gemini()` — loops on `websocket.receive_bytes()` and forwards to the model.
- `gemini_to_browser()` — loops on `session.receive()` and fans out audio + transcripts.

On `WebSocketDisconnect` the bridge ends; other exceptions are logged and, if
possible, reported to the client as an `error` frame.

---

## 4. Terminal App Pipeline (`app.py`)

### 4.1 Task model

The CLI client (`AudioTutor`) runs five concurrent asyncio tasks sharing two
queues:

```
                 ┌──────────────┐  {data, mime}  ┌──────────────┐
  mic (PyAudio)─►│ listen_mic() │───────────────►│  out_queue   │
                 └──────────────┘                └──────┬───────┘
                                                        │
                                                 ┌──────▼───────┐
                                                 │ send_audio() │─► session.send_realtime_input
                                                 └──────────────┘

  session.receive() ─► receive_audio() ─► audio_in_queue ─► play_audio() ─► speaker (PyAudio)
                                   │
                                   └─ prints transcripts / detects interruption

  _watch_for_silence()  ─► debounced clear of "assistant_speaking" flag
```

| Task | Responsibility |
|------|----------------|
| `listen_mic()` | Read 16 kHz mic chunks; skip them while Kate speaks (if `MUTE_MIC_WHILE_SPEAKING`) |
| `send_audio()` | Drain `out_queue` and stream to the model |
| `receive_audio()` | Pull model responses; queue audio; print transcripts; handle interruptions |
| `play_audio()` | Drain `audio_in_queue` and write to the speaker (24 kHz) |
| `_watch_for_silence()` | Clear the "speaking" flag after a quiet debounce window |

### 4.2 Half-duplex & interruption logic

- **`assistant_speaking` (asyncio.Event)** — set when model audio arrives.
- While set and `MUTE_MIC_WHILE_SPEAKING` is `True`, `listen_mic()` drops mic
  chunks so speaker bleed isn't mistaken for the user interrupting.
- **`SPEAKING_SILENCE_DEBOUNCE_S` (0.35s)** — `_watch_for_silence()` only clears
  the flag after the playback queue stays empty, preventing flicker between
  audio chunks of the same reply.
- On a **genuine** `interrupted` signal, the playback queue is flushed and the
  flag cleared immediately. A normal turn completion does **not** flush the
  queue — playback is allowed to finish.

### 4.3 Lifecycle

```
asyncio.run(tutor.run())
   └─ client.aio.live.connect(model, config)
        ├─ create out_queue / audio_in_queue
        ├─ start _watch_for_silence
        ├─ gather(listen_mic, send_audio, receive_audio, play_audio, watcher)
        └─ on exit: cancel all tasks, pya.terminate()
Ctrl+C → "Session ended."
```

---

## 5. Configuration Reference

| Setting | File(s) | Default | Effect |
|---------|---------|---------|--------|
| `GEMINI_API_KEY` | `.env` | — | Auth for the Gemini client (required) |
| `MODEL` | both | `gemini-3.1-flash-live-preview` | Live model id |
| `SEND_SAMPLE_RATE` | both | `16000` | Mic → model rate (Hz) |
| `RECV_SAMPLE_RATE` / `RECEIVE_SAMPLE_RATE` | both | `24000` | Model → speaker rate (Hz) |
| `SYSTEM_PROMPT` | both | — | Kate's persona and coaching rules |
| `CHUNK_SIZE` | `app.py` | `1024` | PyAudio read buffer (frames) |
| `MUTE_MIC_WHILE_SPEAKING` | `app.py` | `True` | Half-duplex mic muting |
| `SPEAKING_SILENCE_DEBOUNCE_S` | `app.py` | `0.35` | Speaking-flag debounce (s) |
| server host/port | `main.py` | `0.0.0.0:8000` | Uvicorn bind (reload on) |

---

## 6. LiveConnectConfig

Both entry points open the session with the same configuration:

```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],                               # spoken replies
    system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    output_audio_transcription={},                               # transcribe Kate
    input_audio_transcription={},                                # transcribe user
)
```

- `response_modalities=["AUDIO"]` → model replies with speech.
- Transcription options surface text alongside audio so the UI can show a chat
  log and the terminal can print turns.

---

## 7. Sequence Diagram (Web App)

```
User        Browser            FastAPI /ws         Gemini Live
 │   click     │                   │                   │
 │────────────►│  open WS          │                   │
 │             │──────────────────►│  accept           │
 │             │                   │  live.connect ───►│
 │             │                   │  silent pulse ───►│
 │             │                   │                   │  (Kate intro)
 │             │   binary PCM  ◄───│◄──────────────────│  audio
 │  hears Kate │◄──────────────────│                   │
 │             │   JSON kate_txt◄──│◄──────────────────│  output_transcription
 │  speaks     │                   │                   │
 │────────────►│  binary PCM ─────►│  realtime_input ─►│
 │             │   JSON user_txt◄──│◄──────────────────│  input_transcription
 │             │        ...        │        ...        │
 │  close tab  │  close WS ───────►│  session closes   │
```

---

## 8. Deployment (Docker)

```
Dockerfile (python:3.11-slim)
  ├─ apt install: build-essential, portaudio19-dev, libportaudio2
  ├─ pip install -r requirements.txt
  └─ CMD: uvicorn main:app --host 0.0.0.0 --port ${PORT}
```

PortAudio system libs are installed so `pyaudio` builds inside the image even
though the web server itself does not use the local mic/speaker.

```bash
docker build -t ai-tutor .
docker run -p 8000:8000 -e PORT=8000 -e GEMINI_API_KEY=your_key_here ai-tutor
```

---

## 9. Error Handling Summary

| Layer | Failure | Behavior |
|-------|---------|----------|
| Startup | Missing `GEMINI_API_KEY` | `main.py` raises `RuntimeError`; `app.py` prints error and exits |
| Web `/ws` | Client disconnects | Caught as `WebSocketDisconnect`, bridge stops cleanly |
| Web `/ws` | Session/model error | Logged; `error` JSON sent to the browser if possible |
| Browser | Mic permission denied | Error bubble shown in chat |
| Browser | WS error | Status set to `error`; "Connection error" bubble |
| Terminal | Model/connect error | Logged with traceback; PyAudio terminated in `finally` |

---

## 10. Glossary

- **Gemini Live API** — Google's low-latency streaming model API for
  bidirectional audio conversation.
- **PCM** — raw uncompressed audio samples (16-bit signed integers here).
- **Half-duplex** — one side speaks at a time; the mic is muted while Kate talks.
- **Barge-in / interruption** — the user talks over Kate; the model signals
  `interrupted` and queued audio is flushed.
- **Silent pulse** — a frame of zero-value PCM used to trigger the model's first turn.
```
