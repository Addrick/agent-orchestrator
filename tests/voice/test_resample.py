# tests/voice/test_resample.py
import numpy as np

from src.voice.resample import TARGET_RATE, to_16k_mono


def _pcm(samples):
    return np.asarray(samples, dtype=np.int16).tobytes()


def test_empty_returns_empty():
    assert to_16k_mono(b"", 48000, 2) == b""


def test_mono_16k_passthrough():
    data = _pcm([100, -200, 300, -400])
    out = to_16k_mono(data, TARGET_RATE, 1)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [100, -200, 300, -400]


def test_stereo_downmix_averages_channels():
    # L/R interleaved: (10,20),(30,40) -> mono 15,35
    data = _pcm([10, 20, 30, 40])
    out = to_16k_mono(data, TARGET_RATE, 2)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [15, 35]


def test_downsample_48k_to_16k_thirds_the_length():
    # 48k -> 16k is a 3x decimation in rate; output ~ len/3.
    samples = list(range(0, 48))  # 48 mono samples @48k = 1ms
    out = to_16k_mono(_pcm(samples), 48000, 1)
    n = len(np.frombuffer(out, dtype=np.int16))
    assert 14 <= n <= 18  # ~16 samples


def test_trailing_partial_stereo_frame_is_trimmed():
    # Odd sample count with channels=2 must not crash.
    out = to_16k_mono(_pcm([1, 2, 3]), TARGET_RATE, 2)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [1]  # (1,2)->1, drops the 3
