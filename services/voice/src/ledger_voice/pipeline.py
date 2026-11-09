"""Pipecat pipeline for one call: telephony audio in, validated speech out.

    transport in -> VAD -> STT -> EngineAnswer (turn detection + answer) -> TTS -> transport out

There is deliberately no LLM service (ADR 0006). ``assert_speech_path`` enforces
that structure for every pipeline built here, failing the call rather than
running a pipeline that could speak unvalidated text.
"""

from collections.abc import Sequence

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.tts_service import TTSService

from ledger_voice.answer import EngineAnswerProcessor
from ledger_voice.config import VoiceSettings

TELEPHONY_SAMPLE_RATE = 8000


class UnsafePipelineError(RuntimeError):
    """The pipeline could speak text that did not come from the answer step."""


def assert_speech_path(processors: Sequence[FrameProcessor]) -> None:
    if any(isinstance(p, LLMService) for p in processors):
        raise UnsafePipelineError("Voice pipelines must not contain an LLM service")
    tts = [i for i, p in enumerate(processors) if isinstance(p, TTSService)]
    answers = [i for i, p in enumerate(processors) if isinstance(p, EngineAnswerProcessor)]
    if len(tts) != 1 or len(answers) != 1 or answers[0] != tts[0] - 1:
        raise UnsafePipelineError("Exactly one EngineAnswer step must feed the only TTS service")


def speech_services(settings: VoiceSettings) -> tuple[FrameProcessor, TTSService]:
    """ElevenLabs Scribe v2 Realtime and Flash v2.5, both at the telephony rate."""
    settings.require_live()
    assert settings.elevenlabs_api_key is not None
    key = settings.elevenlabs_api_key.get_secret_value()
    stt = ElevenLabsRealtimeSTTService(
        api_key=key,
        sample_rate=TELEPHONY_SAMPLE_RATE,
        settings=ElevenLabsRealtimeSTTService.Settings(
            language=settings.stt_language, keyterms=settings.keyterms() or None
        ),
    )
    tts = ElevenLabsTTSService(
        api_key=key,
        sample_rate=TELEPHONY_SAMPLE_RATE,
        settings=ElevenLabsTTSService.Settings(
            voice=settings.tts_voice_id, model=settings.tts_model
        ),
    )
    return stt, tts


def build_pipeline(
    input_processor: FrameProcessor,
    output_processor: FrameProcessor,
    answer: EngineAnswerProcessor,
    *,
    stt: FrameProcessor,
    tts: TTSService,
) -> Pipeline:
    processors: list[FrameProcessor] = [
        input_processor,
        VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=TELEPHONY_SAMPLE_RATE)),
        stt,
        answer,
        tts,
        output_processor,
    ]
    assert_speech_path(processors)
    return Pipeline(processors)
