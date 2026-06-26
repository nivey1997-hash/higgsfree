"""Kokoro-ONNX TTS wrapper with lazy model loading."""
import io
import os
import logging
import numpy as np

log = logging.getLogger(__name__)

# Model file locations. Default to bare filenames (cwd) for backwards
# compatibility, but allow absolute overrides so the fallback works no matter
# which directory the pipeline is launched from (e.g. Jenkins workspaces).
KOKORO_MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", "kokoro-v0_19.onnx")
KOKORO_VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", "voices.bin")

_model = None
_voices = None


def _load_model():
    global _model, _voices
    if _model is not None:
        return _model, _voices
    log.info(f"Loading Kokoro-ONNX model ({KOKORO_MODEL_PATH})...")
    from kokoro_onnx import Kokoro
    _model = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    log.info("Kokoro model loaded.")
    return _model, _voices


def synthesize(text: str, voice: str = "af_heart") -> bytes:
    """Synthesize text to WAV audio bytes.

    Args:
        text: The text to synthesize.
        voice: Kokoro voice ID.

    Returns:
        WAV audio as bytes.
    """
    import soundfile as sf

    model, _ = _load_model()
    samples, sample_rate = model.create(text, voice=voice, speed=1.0, lang="en-us")

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()
