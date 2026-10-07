"""Rotation in free fall: the operator flying round the pair, the pair spinning, or both
turning together — one class (the annotator found no point in separating them).

A window's features come from the camera gyroscope and from the pair tracked through the
window (tandem.visual.pair_track: pair share of frame, pair-crop and whole-frame DINOv2
embeddings). A logistic regression trained on 170 hand-labelled 8-10 s windows scores
them (weights: tandem/visual/rotation.json); ``detect`` slides it over the free fall.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

FEATURES = ("g_net8", "g_net", "g_path", "g_consist", "g_has",
            "p_seen", "p_med", "p_trend", "p_trend_abs", "p_cv",
            "c_drift", "c_step", "c_span", "f_drift", "f_step", "b_move")
_JSON = os.path.join(os.path.dirname(__file__), "visual", "rotation.json")
WINDOW_S = 8.0
STEP_S = 1.0
_w = None


def _drift(E, ok):
    E = E[ok]
    if len(E) < 3:
        return 0.0, 0.0, 0.0
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim0 = E @ E[0]
    step = 1 - np.sum(E[1:] * E[:-1], axis=1)
    return float(1 - sim0.min()), float(step.mean()), float(1 - E[0] @ E[-1])


def features(t, frac, crop_emb, frame_emb, box, gyro: dict | None, lo: float, hi: float) -> dict:
    """Features of the window [lo, hi] from a pair track (arrays at the track's frames)
    and the gyro dict {fs, t, gx, gy, gz} (or None)."""
    t = np.asarray(t, float); m = (t >= lo) & (t <= hi)
    fr = np.asarray(frac, float)[m]
    out = dict.fromkeys(FEATURES, 0.0)
    if gyro and gyro.get("gx"):
        gt = np.asarray(gyro["t"], float); g = (gt >= lo) & (gt <= hi)
        if g.sum() > 5:
            fs = gyro["fs"]
            G = np.array([gyro["gx"], gyro["gy"], gyro["gz"]], float).T[g]
            cum = np.vstack([np.zeros(3), np.cumsum(G / fs, 0)])
            n8 = int(round(min(WINDOW_S, hi - lo) * fs))
            net8 = max(np.linalg.norm(cum[i + n8] - cum[i]) for i in range(0, max(1, len(cum) - n8)))
            out["g_net8"] = math.degrees(net8)
            out["g_net"] = math.degrees(np.linalg.norm(cum[-1]))
            out["g_path"] = math.degrees(np.linalg.norm(G, axis=1).sum() / fs)
            out["g_consist"] = out["g_net"] / max(out["g_path"], 1.0)
            out["g_has"] = 1.0
    if len(fr):
        seen = fr >= 0.02
        out["p_seen"] = float(seen.mean())
        out["p_med"] = float(np.median(fr))
        th = max(1, len(fr) // 3)
        tr = math.log((fr[-th:].mean() + 1e-3) / (fr[:th].mean() + 1e-3))
        out["p_trend"], out["p_trend_abs"] = tr, abs(tr)
        out["p_cv"] = float(fr.std() / (fr.mean() + 1e-3))
        ce = np.asarray(crop_emb, float)[m]; ok = np.linalg.norm(ce, axis=1) > 0
        out["c_drift"], out["c_step"], out["c_span"] = _drift(ce, ok & seen)
        fe = np.asarray(frame_emb, float)[m]
        out["f_drift"], out["f_step"], _ = _drift(fe, np.ones(len(fe), bool))
        bx = [b for b, k in zip(np.asarray(box, dtype=object)[m], seen) if b is not None and k]
        if len(bx) >= 3:
            cx = np.array([(b[0] + b[2]) / 2 for b in bx]); cy = np.array([(b[1] + b[3]) / 2 for b in bx])
            out["b_move"] = float(np.hypot(np.diff(cx), np.diff(cy)).mean())
    return out


def _load():
    global _w
    if _w is None:
        with open(_JSON, encoding="utf-8") as f:
            _w = json.load(f)
    return _w


def score(feat: dict) -> float:
    """Probability that the window shows rotation."""
    w = _load()
    x = (np.array([feat[k] for k in w["features"]], float) - np.array(w["mu"])) / np.array(w["sd"])
    return float(1 / (1 + math.exp(-(x @ np.array(w["coef"]) + w["bias"]))))


def detect(t, frac, crop_emb, frame_emb, box, gyro, lo: float, hi: float) -> list[dict]:
    """Rotation segments in [lo, hi]: WINDOW_S windows every STEP_S scored, those at or
    above the trained threshold merged; each segment {start_s, end_s, score}."""
    w = _load()
    hits = []
    s = lo
    while s + WINDOW_S <= hi + 1e-6:
        p = score(features(t, frac, crop_emb, frame_emb, box, gyro, s, s + WINDOW_S))
        if p >= w["threshold"]:
            hits.append((s, p))
        s += STEP_S
    segs = []
    for s, p in hits:
        if segs and s <= segs[-1]["end_s"]:
            segs[-1]["end_s"] = round(s + WINDOW_S, 2); segs[-1]["score"] = max(segs[-1]["score"], round(p, 3))
        else:
            segs.append({"start_s": round(s, 2), "end_s": round(s + WINDOW_S, 2), "score": round(p, 3)})
    return segs
