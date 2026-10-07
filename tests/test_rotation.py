import math

import numpy as np

from tandem import rotation as R
from tandem import highlights as hl


def _track(n=40, fps=2.0, t0=100.0, frac=0.15, drift=False):
    t = t0 + np.arange(n) / fps
    rng = np.random.default_rng(0)
    base = rng.normal(size=384)
    crop = np.stack([base + (i * 0.5 * rng.normal(size=384) if drift else 0.01 * rng.normal(size=384)) for i in range(n)])
    return t, np.full(n, frac), crop, crop.copy(), [[0.4, 0.4, 0.6, 0.6]] * n


def _gyro(rate_deg, t0=95.0, secs=30, fs=10.0):
    n = int(secs * fs)
    return {"fs": fs, "t": list(t0 + np.arange(n) / fs), "gx": [math.radians(rate_deg)] * n, "gy": [0.0] * n, "gz": [0.0] * n}


def test_features_see_the_turn_and_the_pair():
    t, frac, crop, frame, box = _track()
    f = R.features(t, frac, crop, frame, box, _gyro(45), 100, 108)
    assert 340 < f["g_net8"] < 380 and f["g_has"] == 1.0 and f["g_consist"] > 0.9
    assert f["p_seen"] == 1.0 and abs(f["p_trend"]) < 1e-6
    still = R.features(t, frac, crop, frame, box, _gyro(0), 100, 108)
    assert still["g_net8"] == 0.0


def test_detect_scores_a_turning_window_above_a_still_one():
    t, frac, crop, frame, box = _track(drift=True)
    turning = R.score(R.features(t, frac, crop, frame, box, _gyro(45), 100, 108))
    t2, frac2, crop2, frame2, box2 = _track(drift=False)
    still = R.score(R.features(t2, frac2, crop2, frame2, box2, _gyro(0), 100, 108))
    assert turning > still
    assert R.detect(t, frac, crop, frame, box, _gyro(45), 100, 119)        # finds a segment


def test_highlights_use_rotation_segments_when_given():
    ts = list(range(100, 140)); frac = [0.15] * 40
    got = hl.from_signals(100, 140, ts, frac, None, rotations=[{"start_s": 110, "end_s": 120, "score": 0.9}])
    rot = [h for h in got if h.kind == "orbit"]
    assert len(rot) == 1 and rot[0].source == "rotation" and 110 <= rot[0].start_s <= 112
