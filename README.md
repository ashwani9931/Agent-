# AI Tutor — Real-time English Communication Tutor

A real-time, voice-to-voice English communication tutor powered by Google's
**Gemini Live API**. You speak into your microphone, and an AI consultant named
**"Kate"** listens, responds with natural speech, and gives live feedback on your
grammar, sentence structure, pronunciation, and technical accuracy.

The project ships in two flavors:

| Entry point | Interface | Audio I/O | Best for |
|-------------|-----------|-----------|----------|
| `main.py`   | Web app (FastAPI + browser) | Browser mic/speaker over WebSocket | Deployed / shared use |
| `app.py`    | Command-line (terminal) | Local mic/speaker via PyAudio | Quick local testing |

Both use the same model (`gemini-3.1-flash-live-preview`) and the same system
prompt that defines Kate's persona and coaching behavior.

---

## Features

- **Voice-to-voice conversation** — speak naturally, hear spoken replies with low latency.
- **Kate, the AI consultant** — introduces herself first, then coaches you through a session.
- **Live feedback** — full-sentence grammar reformulations, technical corrections, and pronunciation guidance.
- **Live transcripts** — both your speech and Kate's replies are transcribed and shown in the chat (web UI).
- **Animated web UI** — single-page interface with an animated orb, waveform visualizer, mic level meter, and chat bubbles.
- **Echo handling** — the terminal client mutes the mic while Kate is speaking to avoid feedback loops; the web UI uses browser echo cancellation.
- **Containerized** — a `Dockerfile` is included for deploying the web server.

---

## Architecture

### Web app (`main.py`)

```
Browser (mic)  ──PCM 16 kHz──►  WebSocket /ws  ──►  Gemini Live session
Browser (spkr) ◄─PCM 24 kHz──  WebSocket /ws  ◄──  Gemini Live session
```

- **FastAPI** serves an embedded single-page HTML UI at `GET /` (the full UI is
  inlined in `main.py`, so no separate static build step is required).
- `GET /health` returns a simple status/model JSON for health checks.
- `WS /ws` opens a bridge between the browser and a Gemini Live session:
  - `mic_to_gemini()` forwards raw PCM audio bytes from the browser to the model.
  - `gemini_to_browser()` streams model audio back as binary frames, and sends
    JSON text frames for user/tutor transcripts and interruption events.
- On connect, the server sends a short silent audio pulse to trigger Kate's
  mandatory self-introduction.

> Note: a `static/` folder is mounted at `/static`, but the primary UI is the
> embedded HTML returned by `GET /`.

### Terminal app (`app.py`)

A standalone asyncio client that uses **PyAudio** for local microphone capture
and speaker playback. It runs four concurrent tasks — capture mic, send audio,
receive audio, play audio — plus a debounced "silence watcher" that decides when
Kate has finished speaking. When `MUTE_MIC_WHILE_SPEAKING` is `True` (default),
the mic is muted while Kate talks to prevent speaker bleed from being
misinterpreted as an interruption.

---

## Audio Format

The Gemini Live API expects 16-bit PCM audio:

- **Mic → model:** 16 kHz (`SEND_SAMPLE_RATE`)
- **Model → speaker:** 24 kHz (`RECV_SAMPLE_RATE` / `RECEIVE_SAMPLE_RATE`)

---

## Prerequisites

- **Python 3.10+** (the Dockerfile uses 3.11).
- A **Google Gemini API key** with access to the Live API.
- For the terminal app (`app.py`): working microphone and speakers, plus
  **PortAudio** installed on your system (PyAudio depends on it).
  - macOS: `brew install portaudio`
  - Debian/Ubuntu: `sudo apt-get install portaudio19-dev`

---

## Setup

1. Clone the repository and change into the project directory.

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API key:

   ```env
   GEMINI_API_KEY=your_key_here
   ```

   The `.env` file is git-ignored and must not be committed.

---

## Running

### Web app

```bash
python main.py
```

Then open <http://localhost:8000> in your browser, click **Start Session**, and
allow microphone access. Kate will introduce herself and begin the session.

The server runs with auto-reload enabled on `0.0.0.0:8000`.

### Terminal app

```bash
python app.py
```

Start talking once you see `>> Microphone live.`. Press `Ctrl+C` to end the
session. Using headphones is recommended to avoid echo.

---

## Configuration

Key settings live near the top of each file:

| Setting | File | Default | Purpose |
|---------|------|---------|---------|
| `MODEL` | both | `gemini-3.1-flash-live-preview` | Gemini Live model id |
| `SEND_SAMPLE_RATE` | both | `16000` | Mic → model sample rate (Hz) |
| `RECV_SAMPLE_RATE` / `RECEIVE_SAMPLE_RATE` | both | `24000` | Model → speaker sample rate (Hz) |
| `SYSTEM_PROMPT` | both | — | Defines Kate's persona and coaching rules |
| `MUTE_MIC_WHILE_SPEAKING` | `app.py` | `True` | Half-duplex mic muting to prevent feedback |
| `SPEAKING_SILENCE_DEBOUNCE_S` | `app.py` | `0.35` | Debounce before marking Kate "done speaking" |

If the model id errors, verify the exact Live model name in Google AI Studio.

---

## Docker

Build and run the web server in a container:

```bash
docker build -t ai-tutor .
docker run -p 8000:8000 -e PORT=8000 -e GEMINI_API_KEY=your_key_here ai-tutor
```

The image installs PortAudio system libraries so PyAudio can build, then starts
the app with `uvicorn main:app` bound to the `PORT` environment variable.

---

## Project Structure

```
Agent/
├── app.py              # Terminal voice tutor (PyAudio + Gemini Live)
├── main.py             # FastAPI web server with embedded UI + WebSocket bridge
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build for the web server
├── logo_base64.txt     # Base64-encoded logo asset
├── static/
│   ├── index.html      # Standalone copy of the web UI
│   └── bg.jpg          # Background image
└── .env                # GEMINI_API_KEY (git-ignored, create locally)
```

---

## Dependencies

- `google-genai` — Gemini Live API client
- `fastapi` + `uvicorn[standard]` — web server (web app)
- `python-multipart` — form/multipart support for FastAPI
- `pyaudio` — local microphone/speaker I/O (terminal app)
- `python-dotenv` — loads `GEMINI_API_KEY` from `.env`

---

## Troubleshooting

- **`GEMINI_API_KEY not found`** — ensure a `.env` file exists in the project root with a valid key.
- **Model id error** — confirm the Live model name in Google AI Studio and update `MODEL`.
- **No audio / echo in terminal app** — use headphones, or keep `MUTE_MIC_WHILE_SPEAKING = True`.
- **PyAudio install fails** — install PortAudio system libraries first (see Prerequisites).
- **Web mic not working** — browsers require a secure context; use `localhost` or HTTPS, and grant mic permission.
- **"Connection error" in the web UI** — confirm the server is running on port 8000.

---

## Usage Tips

- Use headphones to prevent echo.
- Speak clearly at a natural pace and finish your thought — Kate listens before responding.
- Ask Kate to roleplay technical interviews, architecture reviews, or demos.
- Grammar and technical errors are corrected live.
