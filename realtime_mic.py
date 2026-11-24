"""Real-time microphone speech translation with voice activity detection."""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import VADIterator, load_silero_vad

from pipeline import LANG_MAP, MultilingualS2STPipeline

LOGGER = logging.getLogger(__name__)

# User-configurable language settings.
SOURCE_LANG = "en"  # en, ml, hi, fr, es
TARGET_NLLB = "mal_Mlym"
TARGET_TTS = "mal"
STOP_PHRASES = ["stop translation", "stop listening", "stop the translation"]

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # Silero VAD expects 512 samples at 16 kHz.
MAX_BLOCKS = int(15 * SAMPLE_RATE / BLOCK_SIZE)

_audio_queue: queue.Queue[np.ndarray] = queue.Queue()
_utterance_queue: queue.Queue[np.ndarray] = queue.Queue()
_speaking = threading.Event()
_stop_event = threading.Event()
pipeline = MultilingualS2STPipeline()


def is_stop_command(text: str) -> bool:
    """Return True when the transcript contains a configured stop phrase."""
    cleaned = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return any(phrase in cleaned for phrase in STOP_PHRASES)


def keyboard_listener() -> None:
    """Allow the user to stop the application by entering q."""
    while not _stop_event.is_set():
        try:
            if input().strip().lower() == "q":
                LOGGER.info("Stop requested from keyboard")
                _stop_event.set()
        except EOFError:
            return


def microphone_callback(indata, frames, timing, status) -> None:
    """Queue microphone audio while playback is inactive."""
    if status:
        LOGGER.warning("Microphone stream status: %s", status)
    if not _speaking.is_set() and not _stop_event.is_set():
        _audio_queue.put(indata[:, 0].copy())


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode a float audio array as in-memory PCM WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def translation_worker() -> None:
    """Process completed utterances sequentially and play translated speech."""
    while not _stop_event.is_set():
        try:
            audio = _utterance_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            started = time.perf_counter()
            transcript, _ = pipeline.transcribe(to_wav_bytes(audio), SOURCE_LANG)
            if not transcript:
                continue
            print(f"\nYou:        {transcript}")
            if is_stop_command(transcript):
                LOGGER.info("Voice stop command detected")
                _stop_event.set()
                break

            translated = pipeline.translate(
                transcript, LANG_MAP.get(SOURCE_LANG, "eng_Latn"), TARGET_NLLB
            )
            print(f"Translated: {translated}")
            audio_bytes, _ = pipeline.synthesize(translated, TARGET_TTS)
            LOGGER.info("Translation ready in %.1fs", time.perf_counter() - started)

            waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            _speaking.set()
            sd.play(waveform, sample_rate)
            while sd.get_stream().active and not _stop_event.is_set():
                time.sleep(0.05)
            sd.stop()
        except Exception:
            LOGGER.exception("Speech translation failed")
        finally:
            with _audio_queue.mutex:
                _audio_queue.queue.clear()
            _speaking.clear()
            _utterance_queue.task_done()


def main() -> None:
    """Start the microphone, VAD, translation worker and keyboard listener."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    pipeline.warmup(SOURCE_LANG, TARGET_NLLB, TARGET_TTS)
    vad = VADIterator(
        load_silero_vad(), sampling_rate=SAMPLE_RATE, threshold=0.5,
        min_silence_duration_ms=450, speech_pad_ms=200,
    )
    threading.Thread(target=translation_worker, daemon=True, name="translation-worker").start()
    threading.Thread(target=keyboard_listener, daemon=True, name="keyboard-listener").start()

    preroll: deque[np.ndarray] = deque(maxlen=8)
    buffer: list[np.ndarray] = []
    active = False
    LOGGER.info("Listening: %s -> %s", SOURCE_LANG, TARGET_TTS)
    print("Speak a sentence, then pause. Say 'stop translation' or type q + Enter to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
            dtype="float32", callback=microphone_callback,
        ):
            while not _stop_event.is_set():
                try:
                    chunk = _audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                event = vad(torch.from_numpy(chunk))
                if event and "start" in event:
                    active, buffer = True, list(preroll)
                if active:
                    buffer.append(chunk)
                else:
                    preroll.append(chunk)
                ended = event and "end" in event
                if active and (ended or len(buffer) > MAX_BLOCKS):
                    active = False
                    if len(buffer) * BLOCK_SIZE / SAMPLE_RATE > 0.4:
                        _utterance_queue.put(np.concatenate(buffer))
                    buffer = []
                    vad.reset_states()
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received")
    finally:
        _stop_event.set()
        sd.stop()
        LOGGER.info("Microphone translator stopped")


if __name__ == "__main__":
    main()
