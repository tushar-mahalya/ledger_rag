import asyncio

import numpy as np
import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pydantic import SecretStr

from ledger_voice.answer import EngineAnswerProcessor
from ledger_voice.config import VoiceSettings
from ledger_voice.engine import Reply
from ledger_voice.logs import configure_logging
from ledger_voice.pipeline import (
    UnsafePipelineError,
    assert_speech_path,
    build_pipeline,
    speech_services,
)
from ledger_voice.simulate import mulaw_decode, mulaw_encode


class FakeEngine:
    def __init__(self, text: str = "Revenue was 120 million.", delay: float = 0.0) -> None:
        self.text, self.delay, self.questions = text, delay, []  # type: list[str]

    async def answer(self, transcript: str) -> Reply:
        self.questions.append(transcript)
        await asyncio.sleep(self.delay)
        return Reply(self.text, "answered", 1.0, "r1")


def answerer(engine: FakeEngine, **kwargs: object) -> EngineAnswerProcessor:
    """Turns driven by proposals, as a turn model would propose them in production."""
    return EngineAnswerProcessor(
        engine, user_turn_strategies=ExternalUserTurnStrategies(), **kwargs
    )  # type: ignore[arg-type]


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="t", finalized=True)


async def test_a_finished_turn_speaks_only_the_engine_answer() -> None:
    engine = FakeEngine()
    replies: list[tuple[str, Reply]] = []

    async def hook(transcript: str, reply: Reply) -> None:
        replies.append((transcript, reply))

    processor = answerer(engine, on_reply=hook)
    down, _ = await run_test(
        processor,
        frames_to_send=[
            ProposedUserStartedSpeakingFrame(),
            transcription("What was Aurora revenue"),
            transcription("in 2024?"),
            LLMTextFrame("an unvalidated model token"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.8),
        ],
    )
    assert engine.questions == ["What was Aurora revenue in 2024?"]
    spoken = [frame for frame in down if isinstance(frame, TTSSpeakFrame)]
    assert [frame.text for frame in spoken] == ["Revenue was 120 million."]
    # Transcripts and any other text never travel on towards TTS.
    assert not any(isinstance(frame, (TranscriptionFrame, LLMTextFrame)) for frame in down)
    assert replies[0][0] == "What was Aurora revenue in 2024?"


async def test_noise_turns_do_not_reach_the_engine() -> None:
    engine = FakeEngine()
    down, _ = await run_test(
        answerer(engine),
        frames_to_send=[
            ProposedUserStartedSpeakingFrame(),
            transcription("um"),
            ProposedUserStoppedSpeakingFrame(),
        ],
    )
    assert engine.questions == [] and not any(isinstance(f, TTSSpeakFrame) for f in down)


async def test_barge_in_cancels_the_pending_answer() -> None:
    engine = FakeEngine(delay=1.5)
    processor = answerer(engine)
    down, _ = await run_test(
        processor,
        frames_to_send=[
            ProposedUserStartedSpeakingFrame(),
            transcription("What was revenue in 2024?"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.8),
            InterruptionFrame(),
            SleepFrame(sleep=1.6),
        ],
    )
    assert engine.questions == ["What was revenue in 2024?"]
    assert not any(isinstance(frame, TTSSpeakFrame) for frame in down)
    assert not processor.pending


async def test_transcript_length_is_bounded() -> None:
    engine = FakeEngine()
    await run_test(
        answerer(engine, max_transcript_chars=20),
        frames_to_send=[
            ProposedUserStartedSpeakingFrame(),
            transcription("What was the revenue of Aurora Devices"),
            transcription("in fiscal year 2024 please"),
            ProposedUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.8),
        ],
    )
    assert engine.questions and len(engine.questions[0]) <= 20


def live_settings() -> VoiceSettings:
    return VoiceSettings(
        engine_token=SecretStr("t"),
        elevenlabs_api_key=SecretStr("el-test-key"),
        tts_voice_id="JBFqnCBsd6RMkjVDRZzb",
        stream_secret=SecretStr("s" * 32),
    )


def test_built_pipeline_has_no_llm_and_answer_feeds_tts() -> None:
    stt, tts = speech_services(live_settings())
    answer = EngineAnswerProcessor(FakeEngine())
    pipeline = build_pipeline(FrameProcessor(), FrameProcessor(), answer, stt=stt, tts=tts)
    names = [type(p).__name__ for p in pipeline.processors]
    assert "EngineAnswerProcessor" in names and not any("LLM" in name for name in names)


def test_speech_path_guard_rejects_llm_and_misordered_pipelines() -> None:
    from pipecat.services.llm_service import LLMService

    tts = ElevenLabsTTSService(api_key="k", settings=ElevenLabsTTSService.Settings(voice="v"))
    answer = EngineAnswerProcessor(FakeEngine())

    class FakeLLM(LLMService):  # type: ignore[misc]
        pass

    with pytest.raises(UnsafePipelineError, match="LLM"):
        assert_speech_path([FrameProcessor(), FakeLLM(), answer, tts])
    with pytest.raises(UnsafePipelineError, match="EngineAnswer"):
        assert_speech_path([answer, FrameProcessor(), tts])
    with pytest.raises(UnsafePipelineError, match="EngineAnswer"):
        assert_speech_path([FrameProcessor(), tts])


def test_live_services_refuse_to_start_without_credentials() -> None:
    with pytest.raises(RuntimeError, match="LEDGER_VOICE_ELEVENLABS_API_KEY"):
        speech_services(VoiceSettings(engine_token=SecretStr("t")))


def test_logging_never_drops_to_debug() -> None:
    from loguru import logger

    configure_logging()
    assert logger._core.min_level >= 20  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        configure_logging("DEBUG")


def test_mulaw_round_trip_is_close() -> None:
    signal = (np.sin(np.linspace(0, 40, 1600)) * 12000).astype(np.int16)
    decoded = mulaw_decode(mulaw_encode(signal))
    assert decoded.shape == signal.shape
    assert np.max(np.abs(decoded.astype(int) - signal.astype(int))) < 600


def test_answer_step_keeps_the_turn_processor_lifecycle_hooks() -> None:
    from pipecat.turns.user_turn_processor import UserTurnProcessor

    # Shadowing these would break turn-controller shutdown on cancel/end.
    for name in ("_cancel", "_stop", "_stop_controllers", "_cleanup"):
        assert getattr(EngineAnswerProcessor, name) is getattr(UserTurnProcessor, name), name
