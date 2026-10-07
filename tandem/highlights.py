"""Free-fall highlights: the clips worth cutting between the drogue throw and deploy.

Four kinds of moment, each from the cheapest signal that measures it:

- ``close`` / ``wide`` — shot scale: the share of the frame the tandem pair fills
  (union box of the overlapping person detections). Wide: the pair is visible but small,
  the landscape carries the shot; close: the pair fills the frame.
- ``orbit`` — the operator flying around the pair: the camera turns through a large
  angle around one axis while the pair stays in frame. Measured by the camera gyroscope
  (net rotation over a sliding window, so head bobbing cancels and a sustained turn
  accumulates); ``orbit_from_flow`` gives the same angle from the background's motion
  in the image for files without a gyroscope.
- ``face`` — the pair so close that faces are large; ranked by the camera's smile score
  where the camera writes one (GoPro FACE). The GoPro face box itself is not used: on
  free-fall footage its size does not match the faces in the frame.

Everything here is pure: per-second / per-sample arrays in, ``Highlight`` list out.
Thresholds were set on 231 labelled jumps — see docs/2026-10-07-freefall-highlights.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

# Shot scale: share of the frame filled by the pair's box.
PAIR_SEEN = 0.02      # below this the pair is lost / a speck
WIDE_MAX = 0.10       # <= 10 %: general (wide) shot
CLOSE_MIN = 0.30      # >= 30 %: close-up
FACE_MIN = 0.45       # >= 45 %: faces are large — face close-up territory
CLOSE_MIN_S = 2.0     # shortest run that counts as a close-up / face moment
WIDE_MIN_S = 3.0      # shortest run that counts as a wide shot

# Orbit: net camera rotation over a sliding window, with the pair in frame.
ORBIT_WINDOW_S = 8.0
ORBIT_MIN_DEG = 270.0
ORBIT_PAIR_SHARE = 0.75   # pair visible in >= 75 % of the window's seconds

# Clip lengths and the mix the selector aims for.
CLIP_S = {"face": 3.0, "close": 4.0, "wide": 4.0}
PRIORITY = ("face", "orbit", "close", "wide")
MAX_PER_KIND = {"face": 1, "orbit": 1, "close": 2, "wide": 1}
GAP_S = 0.5


@dataclass
class Highlight:
    kind: str          # face | orbit | close | wide
    start_s: float
    end_s: float
    score: float       # comparable within a kind only
    source: str        # what measured it: detector | gyro | flow

    def as_dict(self) -> dict:
        return asdict(self)


def shot_scale(frac: float) -> str:
    """Pair share of the frame -> none | wide | medium | close."""
    if frac < PAIR_SEEN:
        return "none"
    if frac <= WIDE_MAX:
        return "wide"
    if frac < CLOSE_MIN:
        return "medium"
    return "close"


def _runs(mask: list[bool]) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i)); start = None
    return out


def _clip(start: float, end: float, length: float, lo: float, hi: float) -> tuple[float, float]:
    """A clip of ``length`` centred on [start, end], kept inside [lo, hi]."""
    if end - start <= length:
        return max(lo, start), min(hi, end)
    mid = (start + end) / 2
    a = max(lo, mid - length / 2)
    return a, min(hi, a + length)


def scale_moments(t: list[float], frac: list[float], lo: float, hi: float,
                  smile: list[float] | None = None) -> list[Highlight]:
    """Close-up, face and wide-shot candidates from per-second pair fractions in [lo, hi].

    ``t`` is in seconds at ~1 Hz; ``smile`` (0..100, optional) is the camera's smile
    score at the same instants and ranks face moments."""
    pts = [(ti, f, (smile[i] if smile else 0.0)) for i, (ti, f) in enumerate(zip(t, frac)) if lo <= ti <= hi]
    if not pts:
        return []
    dt = (pts[-1][0] - pts[0][0]) / max(1, len(pts) - 1) if len(pts) > 1 else 1.0
    out: list[Highlight] = []
    for kind, test, min_s in (("face", lambda f: f >= FACE_MIN, CLOSE_MIN_S),
                              ("close", lambda f: f >= CLOSE_MIN, CLOSE_MIN_S),
                              ("wide", lambda f: PAIR_SEEN <= f <= WIDE_MAX, WIDE_MIN_S)):
        for a, b in _runs([test(p[1]) for p in pts]):
            s0, s1 = pts[a][0], pts[b - 1][0] + dt
            if s1 - s0 < min_s:
                continue
            seg = pts[a:b]
            if kind == "face":
                score = max(p[2] for p in seg) + 10 * (s1 - s0)
            elif kind == "close":
                score = sum(p[1] for p in seg) / len(seg) * (s1 - s0)
            else:
                score = s1 - s0
            c0, c1 = _clip(s0, s1, CLIP_S[kind], lo, hi)
            out.append(Highlight(kind, round(c0, 2), round(c1, 2), round(score, 2), "detector"))
    return out


def net_rotation_deg(rates: list[tuple[float, float, float]], fs: float) -> list[float]:
    """For each window of ORBIT_WINDOW_S starting at sample i: |integral of the angular
    velocity vector| in degrees. A sustained turn adds up; oscillation cancels."""
    n = int(round(ORBIT_WINDOW_S * fs))
    if len(rates) <= n:
        return []
    cx = cy = cz = 0.0
    cum = [(0.0, 0.0, 0.0)]
    for gx, gy, gz in rates:
        cx += gx / fs; cy += gy / fs; cz += gz / fs
        cum.append((cx, cy, cz))
    return [math.degrees(math.sqrt(sum((cum[i + n][k] - cum[i][k]) ** 2 for k in range(3))))
            for i in range(len(cum) - n)]


def orbit_moments(t: list[float], rotation_deg: list[float], lo: float, hi: float,
                  pair_seen: callable, source: str) -> list[Highlight]:
    """Orbit candidates: windows [t_i, t_i + ORBIT_WINDOW_S] inside [lo, hi] whose net
    rotation reaches ORBIT_MIN_DEG while ``pair_seen(t0, t1)`` (share of seconds with the
    pair in frame) is at least ORBIT_PAIR_SHARE. Overlapping hits merge; the clip is the
    strongest window of each merged run."""
    hits = []
    for ti, deg in zip(t, rotation_deg):
        if ti < lo or ti + ORBIT_WINDOW_S > hi or deg < ORBIT_MIN_DEG:
            continue
        if pair_seen(ti, ti + ORBIT_WINDOW_S) < ORBIT_PAIR_SHARE:
            continue
        hits.append((ti, deg))
    out, run = [], []
    for h in hits + [(math.inf, 0.0)]:
        if run and h[0] - run[-1][0] > ORBIT_WINDOW_S / 2:
            best = max(run, key=lambda x: x[1])
            out.append(Highlight("orbit", round(best[0], 2), round(best[0] + ORBIT_WINDOW_S, 2),
                                 round(best[1], 1), source))
            run = []
        if h[0] != math.inf:
            run.append(h)
    return out


def flow_rotation_deg(dx_px: list[float], fps: float, width_px: int, hfov_deg: float) -> list[float]:
    """Net horizontal background shift over ORBIT_WINDOW_S as a camera-turn angle:
    pixels -> degrees through the lens' horizontal field of view (linear approximation)."""
    n = int(round(ORBIT_WINDOW_S * fps))
    deg = [d * hfov_deg / width_px for d in dx_px]
    cum = [0.0]
    for v in deg:
        cum.append(cum[-1] + v)
    return [abs(cum[i + n] - cum[i]) for i in range(len(cum) - n)] if len(cum) > n else []


def from_signals(lo: float, hi: float, ts: list[float], frac: list[float], sig=None,
                 rotations: list[dict] | None = None) -> list[Highlight]:
    """Highlights of the free fall [lo, hi] (drogue throw -> deploy) from per-second pair
    fractions ``(ts, frac)`` and, when given, the camera telemetry ``sig`` (gyroscope for
    orbits, GoPro smile score for ranking face moments). ``rotations`` — segments from
    tandem.rotation.detect (orbit, pair spin or carousel, one class) — replace the
    gyro-only orbit rule when given: on 170 labelled windows the trained detector is
    92 % precise at 78 % recall vs 84 % / 79 % for the gyro alone."""
    smile = None
    if sig is not None and getattr(sig, "smile", None) and len(sig.smile) == len(sig.t_s):
        st, sv = sig.t_s, sig.smile
        smile = []
        for t in ts:                                  # max smile within +-0.5 s of each sample
            vals = [v for tt, v in zip(st, sv) if abs(tt - t) <= 0.5]
            smile.append(max(vals) if vals else 0.0)
    cands = scale_moments(list(ts), list(frac), lo, hi, smile)
    if rotations is not None:
        for r in rotations:
            if r["start_s"] >= lo and r["end_s"] <= hi:
                c0, c1 = _clip(r["start_s"], r["end_s"], ORBIT_WINDOW_S, lo, hi)
                cands.append(Highlight("orbit", round(c0, 2), round(c1, 2), round(r["score"], 3), "rotation"))
    elif sig is not None and getattr(sig, "gx", None) and len(sig.gx) == len(sig.t_s):
        def seen(a, b):
            v = [f for t, f in zip(ts, frac) if a <= t <= b]
            return sum(f >= PAIR_SEEN for f in v) / len(v) if v else 0.0
        rot = net_rotation_deg(list(zip(sig.gx, sig.gy, sig.gz)), sig.fs)
        cands += orbit_moments(list(sig.t_s[:len(rot)]), rot, lo, hi, seen, "gyro")
    return select(cands)


def from_file(path: str, drogue_s: float, deploy_s: float, sig=None) -> list[Highlight]:
    """Highlights of one recording's free fall. Runs the shot-scale head at 1 fps over
    [drogue, deploy]; ``sig`` (``build_signals_from_file``) adds orbits and smile ranking.
    Returns [] when the shot-scale head or the backbone is unavailable."""
    from tandem.visual.shot_scale import pair_fractions
    got = pair_fractions(path, drogue_s, deploy_s)
    if got is None:
        return []
    ts, frac = got
    return from_signals(drogue_s, deploy_s, [float(t) for t in ts], [float(f) for f in frac], sig)


def select(cands: list[Highlight]) -> list[Highlight]:
    """Pick the clips to cut: best-first within each kind, kinds in PRIORITY order, at
    most MAX_PER_KIND each, no two clips closer than GAP_S. Returned in time order."""
    chosen: list[Highlight] = []
    for kind in PRIORITY:
        n = 0
        for h in sorted((c for c in cands if c.kind == kind), key=lambda c: -c.score):
            if n >= MAX_PER_KIND[kind]:
                break
            if any(h.start_s < o.end_s + GAP_S and o.start_s < h.end_s + GAP_S for o in chosen):
                continue
            chosen.append(h); n += 1
    return sorted(chosen, key=lambda h: h.start_s)
