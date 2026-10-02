import math

import numpy as np

from spikes.s1_capture import RATE, dbfs, lag_ms, to_pcm16


def test_to_pcm16_scales_and_clips():
    pcm = np.frombuffer(to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0])), "<i2")
    assert pcm.tolist() == [0, 32767, -32767, 32767, -32767]


def test_lag_ms():
    assert lag_ms(RATE, 1.0) == 0
    assert lag_ms(RATE // 2, 1.0) == 500  # half a second of audio missing


def test_dbfs():
    assert dbfs(np.ones(100, dtype="float32")) == 0
    assert dbfs(np.zeros(100, dtype="float32")) == -math.inf
