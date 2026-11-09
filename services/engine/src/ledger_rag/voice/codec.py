"""Pure-Python telephony audio helpers (no ``audioop``; it was removed in Python 3.13).

All PCM is signed 16-bit little-endian mono ("s16le").
"""

import io
import math
import struct
import sys
import wave
from array import array

TELEPHONY_RATES = (8000, 16000, 24000)
MIN_RATE = 8000
MAX_RATE = 48000
_BIAS = 0x84
_CLIP = 32635
_LITTLE = sys.byteorder == "little"


def _encode_sample(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(-sample if sample < 0 else sample, _CLIP) + _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not magnitude & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _decode_byte(value: int) -> int:
    value = ~value & 0xFF
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return -magnitude if sign else magnitude


# Indexed by the unsigned 16-bit representation of the sample.
_ENCODE = bytes(_encode_sample(value - 65536 if value > 32767 else value) for value in range(65536))
_DECODE = array("h", (_decode_byte(value) for value in range(256)))


def pcm16_to_samples(pcm: bytes) -> "array[int]":
    """Decode s16le bytes; a trailing odd byte is ignored."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not _LITTLE:
        samples.byteswap()
    return samples


def samples_to_pcm16(samples: "array[int] | list[int]") -> bytes:
    data = samples if isinstance(samples, array) else array("h", samples)
    if _LITTLE:
        return data.tobytes()
    swapped = array("h", data)
    swapped.byteswap()
    return swapped.tobytes()


def mulaw_encode(pcm: bytes) -> bytes:
    """Encode s16le PCM to G.711 mu-law (one byte per sample)."""
    table = _ENCODE
    return bytes(table[sample & 0xFFFF] for sample in pcm16_to_samples(pcm))


def mulaw_decode(data: bytes) -> bytes:
    """Decode G.711 mu-law to s16le PCM."""
    table = _DECODE
    return samples_to_pcm16(array("h", (table[value] for value in data)))


def rms(pcm: bytes) -> float:
    samples = pcm16_to_samples(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _check_rate(rate: int) -> None:
    if not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"Unsupported sample rate {rate}")


class Resampler:
    """Streaming linear-interpolation resampler (8/16/24 kHz, any rate in 8-48 kHz).

    Downsampling applies a short boxcar pre-filter to reduce aliasing; this is
    telephony-grade, not studio-grade, resampling. Odd trailing bytes are carried
    to the next chunk so arbitrary network chunking is safe.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        _check_rate(source_rate)
        _check_rate(target_rate)
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._step = source_rate / target_rate
        self._width = math.ceil(self._step) if source_rate > target_rate else 1
        self._history: list[int] = []
        self._previous: int | None = None
        self._position = 0.0
        self._carry = b""

    def _filter(self, samples: "array[int]") -> list[int]:
        if self._width == 1:
            return list(samples)
        width = self._width
        joined = self._history + list(samples)
        offset = len(self._history)
        total = sum(joined[max(0, offset - width) : offset])
        filtered = []
        for index in range(offset, len(joined)):
            total += joined[index]
            start = index - width + 1
            if start > 0:
                total -= joined[start - 1]
            filtered.append(total // min(width, index + 1))
        self._history = joined[-(width - 1) :]
        return filtered

    def process(self, pcm: bytes) -> bytes:
        pcm = self._carry + pcm
        self._carry = pcm[-1:] if len(pcm) % 2 else b""
        samples = pcm16_to_samples(pcm)
        if self.source_rate == self.target_rate:
            return samples_to_pcm16(samples)
        filtered = self._filter(samples)
        buffer = filtered if self._previous is None else [self._previous, *filtered]
        if not buffer:
            return b""
        output: list[int] = []
        position = self._position
        last = len(buffer) - 1
        step = self._step
        while position < last:
            index = int(position)
            left = buffer[index]
            output.append(round(left + (buffer[index + 1] - left) * (position - index)))
            position += step
        self._position = position - last
        self._previous = buffer[-1]
        return samples_to_pcm16(output)


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """One-shot resampling; output is exactly ``n * target // source`` samples."""
    _check_rate(source_rate)
    _check_rate(target_rate)
    samples = pcm16_to_samples(pcm)
    if source_rate == target_rate or not samples:
        return samples_to_pcm16(samples)
    filtered = Resampler(source_rate, target_rate)._filter(samples)
    count = len(filtered) * target_rate // source_rate
    step = source_rate / target_rate
    last = len(filtered) - 1
    output = []
    for item in range(count):
        position = item * step
        index = min(int(position), last)
        left = filtered[index]
        right = filtered[min(index + 1, last)]
        output.append(round(left + (right - left) * (position - index)))
    return samples_to_pcm16(output)


def build_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm[: len(pcm) - (len(pcm) % 2)])
    return buffer.getvalue()


def parse_wav(data: bytes) -> tuple[bytes, int]:
    """Return mono s16le PCM and the sample rate from a complete PCM WAV file."""
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("Invalid WAV audio") from exc
    if width != 2 or channels not in {1, 2}:
        raise ValueError("Only 16-bit mono or stereo PCM WAV is supported")
    return (_downmix(frames) if channels == 2 else frames), rate


def _downmix(pcm: bytes) -> bytes:
    samples = pcm16_to_samples(pcm)
    return samples_to_pcm16(
        [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples) - 1, 2)]
    )


class WavStreamParser:
    """Incrementally strips a RIFF/WAVE header from a streamed response.

    Streaming TTS servers often write a placeholder (0 or 0xFFFFFFFF) data size,
    so the data chunk is treated as running until the end of the stream.
    """

    max_header_bytes = 8192

    def __init__(self) -> None:
        self._header = bytearray()
        self.sample_rate: int | None = None
        self.channels = 1
        self._carry = b""

    def feed(self, chunk: bytes) -> bytes:
        if self.sample_rate is None:
            self._header.extend(chunk)
            if len(self._header) > self.max_header_bytes:
                raise ValueError("WAV header too large")
            payload = self._parse_header()
            if payload is None:
                return b""
            chunk = payload
        block = 2 * self.channels
        data = self._carry + chunk
        cut = len(data) - (len(data) % block)
        self._carry = data[cut:]
        return _downmix(data[:cut]) if self.channels == 2 else data[:cut]

    def _parse_header(self) -> bytes | None:
        header = bytes(self._header)
        if len(header) < 12:
            return None
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("Invalid WAV stream")
        offset = 12
        rate: int | None = None
        while len(header) >= offset + 8:
            chunk_id = header[offset : offset + 4]
            size = struct.unpack_from("<I", header, offset + 4)[0]
            body = offset + 8
            if chunk_id == b"data":
                if rate is None:
                    raise ValueError("WAV data before format chunk")
                self.sample_rate = rate
                return header[body:]
            if len(header) < body + size:
                return None
            if chunk_id == b"fmt ":
                if size < 16:
                    raise ValueError("Invalid WAV format chunk")
                fmt, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", header, body)
                if fmt not in {1, 0xFFFE} or bits != 16 or channels not in {1, 2}:
                    raise ValueError("Only 16-bit PCM WAV is supported")
                self.channels = channels
            offset = body + size + (size % 2)
        return None
