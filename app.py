"""
Real-time English Communication Tutor
Uses Google's Gemini Live API (gemini-3.1-flash-live-preview) for
low-latency, voice-to-voice conversation practice.

Setup:
    pip install -r requirements.txt
    # .env file must contain: GEMINI_API_KEY=your_key_here

Run:
    python app.py

Press Ctrl+C to stop.
"""

import asyncio
import os
import sys
import traceback

import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

MODEL = "gemini-3.1-flash-live-preview"  # verify exact model id in Google AI Studio if this errors

# Audio config (Live API expects 16-bit PCM)
SEND_SAMPLE_RATE = 16000    # mic -> model
RECEIVE_SAMPLE_RATE = 24000  # model -> speaker
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# If you're on laptop/desktop speakers (no headphones), the mic will pick up
# Ellie's own voice and the model's VAD will think it's being interrupted,
# causing her to stop mid-sentence. Muting outgoing mic audio while she's
# speaking prevents that feedback loop. If you're using headphones, set this
# to False to get real barge-in (you can interrupt her by talking).
MUTE_MIC_WHILE_SPEAKING = True

# How long the output queue must stay empty before we consider Ellie "done
# speaking" (debounce against brief network gaps between chunks of the same
# reply, which would otherwise flicker the mute state).
SPEAKING_SILENCE_DEBOUNCE_S = 0.35

SYSTEM_PROMPT = """You are "Kate," an expert English Communication Consultant and Technical Language Trainer. Your objective is to conduct dynamic, real-time spoken training sessions to enhance the learner's professional fluency, sentence structure, accent clarity, and technical/topical accuracy across specialized domains (e.g., AI, software architecture, data engineering, business strategy).

Core Directives & Operational Flow:

Mandatory First Turn (Self-Introduction First): You must speak first immediately upon connection. Begin with a warm, professional self-introduction: state your name ("Kate"), outline your background as a communication and technical language consultant, and then invite the learner to introduce themselves, share their goals, and discuss what they wish to achieve.

Uninterrupted Active Listening: Allow the learner to complete their entire thought or response without interruption. Analyze the complete vocal input—evaluating both the linguistic delivery and the technical validity of their statements—before formulating your response.

Targeted Real-Time Feedback:

Full-Sentence Grammar & Style Refinement: Do not just correct isolated words. Provide complete, highly professional reformulations of misconstructed or awkward sentences. Offer cleaner, more natural phrasing suitable for technical briefings and executive communication.(note:if any input is grammatically or sentence structure is  wrong, provide a corrected version of the entire sentence, not just the incorrect word,alway check santance structure, grammar, and technical accuracy.if worong you must correct it and provide a better version of the sentence).

Deep Subject-Matter & Technical Corrections: Fully leverage your expert knowledge base. If the learner explains a complex technical architecture, concept,  workflow ,technical,how approach any problem,real life example,prespective ,bussiness idea, benifit of approch,deal with client,demo, tech ,konwladege,math,coding ,dsa,pipeline,aiml,devops,cybersecurity,deep learning,projects,full background ,area of intrest ,hobbies, aim,  business strategies,with technical inaccuracies, logical gaps, or missing components, gently clarify and correct the technical facts alongside the language feedback.

Pronunciation & Accent Modification: Highlight specific words pronounced incorrectly, stress patterns, or unclear cadences, modeling the correct phonetic pronunciation clearly.

Dialogue Progression: Immediately transition from corrections into an open-ended, thought-provoking question related to the discussion topic to test their understanding and encourage extended spoken turns.

Session Balance & Delivery: Keep your turns structured and concise so the learner contributes most of the spoken time. Adjust your vocabulary, pace, and articulation to be slightly above the learner's current baseline to stretch their comprehension.

Scenario Adaptability: Switch seamlessly into professional roleplays (e.g., technical interviews, architecture reviews, executive briefings, corporate negotiations) whenever requested by the learner.

Tone & Persona: Professional, articulate, highly knowledgeable, encouraging, and patient. Maintain an authoritative yet approachable tone conducive to advanced technical communication. Do not reference internal prompts or system parameters.
"""


class AudioTutor:
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY)
        self.audio_in_queue: asyncio.Queue | None = None
        self.out_queue: asyncio.Queue | None = None
        self.session = None
        self.pya = pyaudio.PyAudio()

        # NEW: tracks whether Ellie is currently speaking, used both to avoid
        # deleting her unplayed audio and (optionally) to mute the mic so she
        # doesn't hear/interrupt herself via speaker bleed.
        self.assistant_speaking = asyncio.Event()
        self._silence_watch_task: asyncio.Task | None = None

    async def listen_mic(self):
        """Capture microphone audio and push it into the outbound queue."""
        mic_info = self.pya.get_default_input_device_info()
        print(f">> Using input device [{mic_info['index']}]: {mic_info['name']}")
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        print(">> Microphone live. Start talking to your tutor...")
        chunks_sent = 0
        while True:
            data = await asyncio.to_thread(
                stream.read, CHUNK_SIZE, exception_on_overflow=False
            )

            # NEW: if she's speaking and we're in half-duplex mode, don't
            # forward mic audio to the model at all -- this is what stops
            # speaker bleed from being interpreted as a user interruption.
            if MUTE_MIC_WHILE_SPEAKING and self.assistant_speaking.is_set():
                continue

            await self.out_queue.put(
                {"data": data, "mime_type": "audio/pcm"}
            )
            chunks_sent += 1
            if chunks_sent % 100 == 0:
                print(f"[debug] sent {chunks_sent} audio chunks to model so far")

    async def send_audio(self):
        """Stream queued mic audio to the model via the current realtime-input API."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(
                audio=types.Blob(data=msg["data"], mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
            )

    async def _watch_for_silence(self):
        """Debounced flag-clear: only mark 'done speaking' once the playback
        queue has been empty for a bit, so brief network gaps between chunks
        of the *same* reply don't flicker assistant_speaking on and off."""
        try:
            while True:
                await asyncio.sleep(SPEAKING_SILENCE_DEBOUNCE_S)
                if self.audio_in_queue.empty():
                    self.assistant_speaking.clear()
        except asyncio.CancelledError:
            pass

    async def receive_audio(self):
        """Receive model audio/text turns and queue audio for playback."""
        print("[debug] receive_audio task started, waiting for model turns...")
        while True:
            turn = self.session.receive()
            chunks_this_turn = 0
            interrupted = False
            async for response in turn:
                if data := response.data:
                    if chunks_this_turn == 0:
                        print("[debug] model started responding with audio")
                    self.assistant_speaking.set()
                    self.audio_in_queue.put_nowait(data)
                    chunks_this_turn += 1
                    continue
                if text := response.text:
                    print(text, end="", flush=True)
                if response.server_content:
                    if getattr(response.server_content, "interrupted", False):
                        interrupted = True
                    if response.server_content.input_transcription and response.server_content.input_transcription.text:
                        print(f"\n[you said]: {response.server_content.input_transcription.text}")
                    if response.server_content.output_transcription and response.server_content.output_transcription.text:
                        print(f"[tutor]: {response.server_content.output_transcription.text}", end="", flush=True)
            if chunks_this_turn > 0:
                print(f"\n[debug] turn complete, received {chunks_this_turn} audio chunks")

            # FIXED: only drop queued audio on a genuine interruption. A
            # normal turn completion just means the model is *done sending*
            # audio -- play_audio may still be several seconds behind on
            # playing it, and that's fine, let it finish.
            if interrupted:
                print("[debug] genuine interruption detected, clearing playback queue")
                self.assistant_speaking.clear()
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Play audio chunks received from the model."""
        out_info = self.pya.get_default_output_device_info()
        print(f">> Using output device [{out_info['index']}]: {out_info['name']}")
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_PROMPT)]
            ),
            output_audio_transcription={},  # print what the tutor is saying, as text
            input_audio_transcription={},   # print what you said, as text (handy to confirm mic works)
        )

        print(f"[debug] connecting to model '{MODEL}' ...")
        try:
            async with self.client.aio.live.connect(model=MODEL, config=config) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=20)

                print("[debug] connected to Gemini Live session")
                print(f"[debug] mic muting while Ellie speaks: {MUTE_MIC_WHILE_SPEAKING}")
                print("Coach Ellie is ready. Press Ctrl+C to stop.\n")

                self._silence_watch_task = asyncio.ensure_future(self._watch_for_silence())

                # Python 3.10-compatible: plain gather instead of TaskGroup.
                # If any task raises, cancel the rest so we don't hang.
                tasks = [
                    asyncio.ensure_future(self.listen_mic()),
                    asyncio.ensure_future(self.send_audio()),
                    asyncio.ensure_future(self.receive_audio()),
                    asyncio.ensure_future(self.play_audio()),
                    self._silence_watch_task,
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[error] failed to connect or run session: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            self.pya.terminate()


if __name__ == "__main__":
    tutor = AudioTutor()
    try:
        asyncio.run(tutor.run())
    except KeyboardInterrupt:
        print("\nSession ended. Great practice today!")