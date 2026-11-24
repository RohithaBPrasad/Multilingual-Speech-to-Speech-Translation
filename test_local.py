"""Run the translation pipeline against a local audio file."""

from pathlib import Path

from pipeline import MultilingualS2STPipeline

INPUT_AUDIO = Path("harvard.wav")
OUTPUT_AUDIO = Path("translated_output.wav")
SOURCE_LANGUAGE = "en"
TARGET_NLLB_LANGUAGE = "eng_Latn"  # Example: mal_Mlym for Malayalam.
TARGET_TTS_LANGUAGE = "eng"       # Example: mal for Malayalam.


def main() -> None:
    if not INPUT_AUDIO.is_file():
        raise FileNotFoundError(
            f"Input audio '{INPUT_AUDIO}' was not found. Place a WAV file in the project root."
        )

    pipeline = MultilingualS2STPipeline()
    result = pipeline.run_pipeline(
        audio_bytes=INPUT_AUDIO.read_bytes(),
        target_nllb_lang=TARGET_NLLB_LANGUAGE,
        tts_lang_code=TARGET_TTS_LANGUAGE,
        source_lang=SOURCE_LANGUAGE,
    )
    OUTPUT_AUDIO.write_bytes(result["audio_bytes"])
    print(f"Transcript: {result['transcript']}")
    print(f"Translation: {result['translation']}")
    print(f"Output saved to: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    main()
