"""FastAPI service exposing HTTP and WebSocket speech translation."""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from pipeline import MultilingualS2STPipeline

LOGGER = logging.getLogger(__name__)
pipeline: MultilingualS2STPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the ML models once when the API process starts."""
    global pipeline
    pipeline = MultilingualS2STPipeline()
    pipeline.warmup("en", "hin_Deva", "hin")
    yield
    pipeline = None


app = FastAPI(
    title="Multilingual Speech-to-Speech Translation API",
    description="HTTP and WebSocket endpoints for multilingual speech translation.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline() -> MultilingualS2STPipeline:
    if pipeline is None:
        raise RuntimeError("Translation pipeline has not been initialized.")
    return pipeline


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Basic process health check."""
    return {"status": "ok", "service": "multilingual-s2st"}


@app.post("/translate-audio", tags=["Translation"])
def translate_audio_endpoint(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("hin_Deva"),
    tts_code: str = Form("hin"),
) -> Response:
    """Translate an uploaded audio file and return synthesized WAV audio."""
    input_bytes = audio_file.file.read()
    result = get_pipeline().run_pipeline(input_bytes, target_lang, tts_code)
    headers = {
        "X-Transcript": urllib.parse.quote(result["transcript"]),
        "X-Translation": urllib.parse.quote(result["translation"]),
    }
    return Response(content=result["audio_bytes"], media_type="audio/wav", headers=headers)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket) -> None:
    """Receive audio frames and return synthesized translated audio frames."""
    await websocket.accept()
    LOGGER.info("WebSocket client connected")
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            if not audio_data:
                continue
            result = await asyncio.to_thread(
                get_pipeline().run_pipeline, audio_data, "hin_Deva", "hin"
            )
            await websocket.send_bytes(result["audio_bytes"])
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected")
    except Exception:
        LOGGER.exception("WebSocket processing failed")
        await websocket.close(code=1011)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
