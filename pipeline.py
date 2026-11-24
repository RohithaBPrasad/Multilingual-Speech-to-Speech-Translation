"""Core speech-to-speech translation pipeline.

Stages: speech recognition (Faster-Whisper), machine translation (NLLB-200),
and speech synthesis (Meta MMS/VITS).
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    VitsModel,
)

LOGGER = logging.getLogger(__name__)

# Model and decoding settings. Keep these defaults aligned with the original project.
WHISPER_SIZE = "small"  # tiny / base / small
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NUM_BEAMS = 2
QUANTIZE_NLLB_ON_CPU = True

LANG_MAP = {
    "en": "eng_Latn",
    "ml": "mal_Mlym",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
}


class MultilingualS2STPipeline:
    """Load and run the ASR -> translation -> TTS pipeline."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(torch.get_num_threads())

        LOGGER.info("Loading speech models on device: %s", self.device)
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.asr = WhisperModel(WHISPER_SIZE, device=self.device, compute_type=compute_type)

        LOGGER.info("Loading translation model: %s", NLLB_MODEL)
        self.translator_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        if self.device == "cpu" and QUANTIZE_NLLB_ON_CPU:
            translator_model = torch.quantization.quantize_dynamic(
                translator_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.translator_model = translator_model.to(self.device).eval()
        self.tts_cache: dict[str, tuple[object, VitsModel]] = {}

    def _get_tts_model(self, lang_code: str):
        """Load a language-specific MMS TTS model once and cache it."""
        if lang_code not in self.tts_cache:
            model_id = f"facebook/mms-tts-{lang_code}"
            LOGGER.info("Loading TTS model: %s", model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(self.device).eval()
            self.tts_cache[lang_code] = (processor, model)
        return self.tts_cache[lang_code]

    def warmup(
        self,
        source_lang: str = "en",
        target_nllb_lang: str = "hin_Deva",
        tts_lang_code: str = "hin",
    ) -> None:
        """Run one small request through each stage to warm up the models."""
        LOGGER.info("Warming up speech pipeline...")
        started = time.perf_counter()
        list(self.asr.transcribe(np.zeros(16000, dtype=np.float32), language=source_lang)[0])
        translated = self.translate(
            "Hello, how are you?", LANG_MAP.get(source_lang, "eng_Latn"), target_nllb_lang
        )
        self.synthesize(translated, tts_lang_code)
        LOGGER.info("Pipeline warm-up completed in %.1fs", time.perf_counter() - started)

    def transcribe(self, audio_bytes: bytes, source_lang: str = "en") -> tuple[str, str]:
        """Convert WAV/audio bytes to text using Faster-Whisper."""
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform).squeeze(0)
            audio = waveform.numpy()

        segments, _ = self.asr.transcribe(
            audio,
            language=source_lang,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return transcript, source_lang

    def translate(self, text: str, source_nllb_lang: str, target_nllb_lang: str) -> str:
        """Translate text between NLLB language identifiers."""
        self.translator_tokenizer.src_lang = source_nllb_lang
        inputs = self.translator_tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos = self.translator_tokenizer.convert_tokens_to_ids(target_nllb_lang)
        with torch.inference_mode():
            tokens = self.translator_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=128,
                num_beams=NUM_BEAMS,
            )
        return self.translator_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )[0].strip()

    def synthesize(self, text: str, tts_lang_code: str) -> tuple[bytes, int]:
        """Generate speech audio bytes using the requested MMS language model."""
        processor, model = self._get_tts_model(tts_lang_code)
        inputs = processor(text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            waveform = model(**inputs).waveform
        audio = waveform.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue(), sample_rate

    def run_pipeline(
        self,
        audio_bytes: bytes,
        target_nllb_lang: str,
        tts_lang_code: str,
        source_lang: str = "en",
    ) -> dict:
        """Run speech recognition, translation and synthesis for one audio input."""
        started = time.perf_counter()
        transcript, detected_lang = self.transcribe(audio_bytes, source_lang=source_lang)
        asr_done = time.perf_counter()
        LOGGER.info("ASR completed in %.2fs", asr_done - started)

        if not transcript:
            return {
                "transcript": "",
                "detected_language": detected_lang,
                "translation": "",
                "audio_bytes": b"",
                "sample_rate": 0,
                "timings": {},
            }

        source_nllb_lang = LANG_MAP.get(source_lang, "eng_Latn")
        translated_text = self.translate(transcript, source_nllb_lang, target_nllb_lang)
        translation_done = time.perf_counter()
        LOGGER.info("Translation completed in %.2fs", translation_done - asr_done)

        output_audio, sample_rate = self.synthesize(translated_text, tts_lang_code)
        finished = time.perf_counter()
        LOGGER.info("TTS completed in %.2fs; total %.2fs", finished - translation_done, finished - started)

        return {
            "transcript": transcript,
            "detected_language": detected_lang,
            "translation": translated_text,
            "audio_bytes": output_audio,
            "sample_rate": sample_rate,
            "timings": {
                "asr": asr_done - started,
                "mt": translation_done - asr_done,
                "tts": finished - translation_done,
                "total": finished - started,
            },
        }
