"""Record a short microphone sample and run speech translation."""

import io

import sounddevice as sd
import soundfile as sf

from pipeline import MultilingualS2STPipeline

RECORDING_SECONDS = 5
SAMPLE_RATE = 16000
SOURCE_LANGUAGE = "en"
TARGET_NLLB_LANGUAGE = "mal_Mlym"
TARGET_TTS_LANGUAGE = "mal"
OUTPUT_AUDIO = "live_translated_output.wav"


def main() -> None:
    pipeline = MultilingualS2STPipeline()
    print(f"Recording for {RECORDING_SECONDS} seconds. Speak clearly...")
    recording = sd.rec(
        int(RECORDING_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
        channels=1, dtype="float32",
    )
    sd.wait()
    print("Recording complete. Translating...")

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, recording, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    result = pipeline.run_pipeline(
        audio_bytes=audio_buffer.getvalue(),
        target_nllb_lang=TARGET_NLLB_LANGUAGE,
        tts_lang_code=TARGET_TTS_LANGUAGE,
        source_lang=SOURCE_LANGUAGE,
    )
    print(f"Detected language: {result['detected_language']}")
    print(f"Transcript: {result['transcript']}")
    print(f"Translation: {result['translation']}")
    with open(OUTPUT_AUDIO, "wb") as output_file:
        output_file.write(result["audio_bytes"])
    print(f"Translated audio saved to: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    main()
