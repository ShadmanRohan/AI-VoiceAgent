"""ONNX-free VAD so remote voice does not deadlock on onnxruntime telemetry SQLite."""

from __future__ import annotations

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams


class VolumeVADAnalyzer(VADAnalyzer):
    """Use volume gate only (confidence always 1). Avoids Silero/onnxruntime."""

    def num_frames_required(self) -> int:
        # ~32ms at 16kHz mono PCM16
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0
