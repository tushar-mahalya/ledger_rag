from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30)


class VoiceMetrics:
    """Voice histograms registered into the per-app ``Metrics.registry`` exactly once.

    Each ``create_app`` lifespan owns a fresh ``CollectorRegistry``, so tests that build
    many apps never hit duplicate-timeseries registration errors.
    """

    def __init__(self, registry: CollectorRegistry) -> None:
        self.sessions = Counter(
            "ledger_voice_sessions_total",
            "Voice sessions by transport and outcome",
            ["transport", "outcome"],
            registry=registry,
        )
        self.active_sessions = Gauge(
            "ledger_voice_active_sessions", "Currently open voice sessions", registry=registry
        )
        self.turns = Counter(
            "ledger_voice_turns_total", "Voice turns by outcome", ["outcome"], registry=registry
        )
        self.barge_ins = Counter(
            "ledger_voice_barge_ins_total", "Caller interruptions of agent audio", registry=registry
        )
        self.stt = Histogram(
            "ledger_voice_stt_seconds",
            "End-of-utterance detection to transcript",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )
        self.tts_first_byte = Histogram(
            "ledger_voice_tts_first_byte_seconds",
            "TTS request to first audio bytes",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )
        self.first_audio = Histogram(
            "ledger_voice_first_audio_seconds",
            "End-of-utterance detection to first agent audio frame (excludes VAD hangover)",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )
        self.turn = Histogram(
            "ledger_voice_turn_seconds",
            "End-of-utterance detection to last agent audio frame sent",
            registry=registry,
            buckets=_LATENCY_BUCKETS,
        )
