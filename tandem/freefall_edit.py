"""The 30-second free-fall edit: exit -> wide / medium / orbit or spin / passenger face ->
deploy, cut from one recording, in time order.

Inputs are what the pipeline already has: the phase boundaries (exit, drogue, deploy),
the per-frame analysis of the free fall (tandem.visual.freefall_frames: RT-DETR pair
share of frame, MediaPipe passenger face with smile / jawOpen, pair-crop embedding), the
rotation detector's segments (tandem.rotation) and the camera gyroscope. ``plan`` returns the edit decision list; ``render`` cuts it with
ffmpeg; ``face_options`` lists every usable passenger-face clip to choose from.

Kinds:
  exit     [exit - 1, drogue throw + 0.5] (without a drogue: [exit - 1, exit + 3])
  wide     the most general shots: the 3 s windows where the visible pair is smallest
           (pair seen in >= 80 % of the frames); two are always asked for     (3 s clip)
  medium   pair >= 10 % of the frame (medium and close shots), >= 2.5 s     (3 s clip)
  face     passenger face >= FACE_H of the frame height, >= 1.5 s; ranked by emotion =
           max(smile, jawOpen) at the peak                                    (2.5 s clip)
           — or exactly the face clips a person picked (``faces``)
  rotation the trained rotation detector (tandem.rotation): an orbit, the pair spinning
           or pair + operator turning together — one class                    (6 s clip)
  Without detector segments the older rules stand in:
  orbit    camera turns >= 270 deg in 8 s (gyro) with the pair in frame       (6 s clip)
  spin     the camera holds (gyro < SPIN_GYRO_MAX deg in 6 s) and the pair keeps its size
           while its look drifts (pair-crop embedding) — the pair turning in front of the
           camera; a growing pair is the operator closing in, not a turn       (4 s)
  freefall filler: plain free-fall pieces when the candidates cannot fill the time
  deploy   [deploy - 1.5, deploy + 2.5]
The middle (drogue -> deploy) runs in time order, or shuffled with ``order="random"``
(a fixed seed per jump); source times never overlap either way.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, asdict

from tandem.highlights import net_rotation_deg, ORBIT_MIN_DEG, ORBIT_WINDOW_S

EDIT_S = 30.0
EXIT_PRE, EXIT_POST = 1.0, 3.0
EXIT_AFTER_DROGUE = 0.5  # the exit clip runs through the drogue throw and ends this much after it
WIDEST_SEEN = 0.8        # a "most general" window needs the pair in >= 80 % of its frames
DEPLOY_PRE, DEPLOY_POST = 1.5, 2.5
PAIR_SEEN, WIDE_MAX, MEDIUM_MAX = 0.02, 0.10, 1.01   # medium: 10 %+ (close-ups too)
FACE_H = 0.06            # passenger face height >= 6 % of the frame height
SPIN_WINDOW_S = 6.0
SPIN_GYRO_MAX = 90.0
SPIN_DRIFT_MIN = 0.35    # 1 - min cosine of pair-crop embeddings to the window's first
SPIN_SIZE_RATIO = 1.4    # pair size: last third vs first third of the window (else an approach)
CLIP = {"wide": 3.0, "medium": 3.0, "face": 2.5, "rotation": 6.0, "orbit": 6.0, "spin": 4.0}
# What the middle of the edit asks for, in priority order: (kind, how many).
# Picked faces first (a person chose them), then rotation (rare — a jump has one or two),
# then the two most general shots (many alternatives: the next widest window steps in).
WANT = (("face", 9), ("rotation", 1), ("wide", 2), ("orbit", 1), ("spin", 1), ("medium", 2),
        ("medium", 2), ("wide", 2))
GAP = 0.5
MIN_CLIP = 2.0
FILL = ("medium", "wide", "face", "rotation", "spin")   # topping up the middle after WANT
MAX_STRETCH = 2.0        # a clip may grow by at most this much to fill time
DEPLOY_EXTRA_MAX = 1.5   # beyond that the deploy clip may run longer only so much
EXIT_EXTRA_MAX = 1.0     # ... and the exit clip start earlier (never back into the cabin)


@dataclass
class Cut:
    kind: str
    start_s: float
    end_s: float
    score: float = 0.0

    @property
    def dur(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        return asdict(self)


def _runs(flags):
    out, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i)); s = None
    return out


def _centered(c: float, length: float, lo: float, hi: float) -> tuple[float, float]:
    a = min(max(lo, c - length / 2), max(lo, hi - length))
    return a, min(hi, a + length)


def _widest(fr: list[dict], dt: float, n: int = 40) -> list[Cut]:
    """The most general shots, best first: CLIP["wide"] windows (starts >= 1 s apart)
    ranked by how small the visible pair is on average (pair seen in >= WIDEST_SEEN of
    the frames). Many overlapping alternatives on purpose — when the widest ones sit
    under a picked face or a rotation, the planner takes the next widest free window."""
    k = max(2, int(round(CLIP["wide"] / dt)))
    wins = []
    for i in range(len(fr) - k + 1):
        seg = fr[i:i + k]
        vis = [f["frac"] for f in seg if f["frac"] >= PAIR_SEEN]
        if len(vis) >= WIDEST_SEEN * k:
            wins.append((sum(vis) / len(vis), seg[0]["t"], seg[-1]["t"] + dt))
    out: list[Cut] = []
    for m, a, b in sorted(wins):
        if all(abs(a - c.start_s) >= 1.0 for c in out):
            out.append(Cut("wide", round(a, 2), round(b, 2), round(1 - m, 4)))
        if len(out) == n:
            break
    return out


def candidates(frames: list[dict], lo: float, hi: float, gyro: dict | None = None,
               emb=None, rotations: list[dict] | None = None,
               faces: list[tuple[float, float]] | None = None) -> list[Cut]:
    """All candidate clips of the free fall [lo, hi] (drogue -> deploy - 1.5).
    ``rotations`` (segments from tandem.rotation.detect) replace the gyro orbit and
    spin rules when given."""
    fr = [f for f in frames if lo <= f["t"] <= hi]
    if not fr:
        return []
    dt = (fr[-1]["t"] - fr[0]["t"]) / max(1, len(fr) - 1) if len(fr) > 1 else 0.5
    out: list[Cut] = []

    def scale_runs(kind, test, min_s, score):
        # a long run gives several clips, spread over it; later ones rank lower
        L = CLIP[kind]
        for a, b in _runs([test(f) for f in fr]):
            s0, s1 = fr[a]["t"], fr[b - 1]["t"] + dt
            if s1 - s0 < min_s:
                continue
            n = max(1, int((s1 - s0 + 2 * GAP) // (L + 2 * GAP)))
            base = score(fr[a:b], s1 - s0)
            for k in range(n):
                mid = s0 + (s1 - s0) * (k + 0.5) / n
                c0, c1 = _centered(mid, L, s0, s1)
                out.append(Cut(kind, round(c0, 2), round(c1, 2), round(base * 0.8 ** k, 3)))

    out += _widest(fr, dt)
    scale_runs("medium", lambda f: WIDE_MAX < f["frac"] < MEDIUM_MAX, 2.5,
               lambda seg, d: d * sum(f["frac"] for f in seg) / len(seg))
    if faces is not None:          # the face clips a person picked, nothing automatic
        for a, b in faces:
            a, b = max(lo, a), min(hi, b)
            if b - a >= MIN_CLIP / 2:
                out.append(Cut("face", round(a, 2), round(b, 2), 1.0))
    # passenger face: clip centred on the most emotional frame of each run
    for a, b in ([] if faces is not None else
                 _runs([bool(f.get("passenger")) and f["passenger"]["h"] >= FACE_H for f in fr])):
        if (b - a) * dt < 1.5:
            continue
        seg = fr[a:b]
        emo = lambda f: max(f["passenger"]["smile"], f["passenger"]["jaw"])
        peak = max(seg, key=emo)
        s0, s1 = seg[0]["t"], seg[-1]["t"] + dt
        if s1 - s0 >= CLIP["face"]:                  # stay inside the face run
            c0, c1 = _centered(peak["t"], CLIP["face"], s0, s1)
        else:
            c0, c1 = _centered((s0 + s1) / 2, CLIP["face"], lo, hi)
        out.append(Cut("face", round(c0, 2), round(c1, 2), round(emo(peak) + peak["passenger"]["h"], 3)))
    if rotations is not None:
        for r in rotations:
            s0, s1 = max(lo, r["start_s"]), min(hi, r["end_s"])
            if s1 - s0 >= MIN_CLIP:
                L = CLIP["rotation"]           # centred, then flush left / right: room around other clips
                for k, mid in enumerate(((s0 + s1) / 2, s0 + L / 2, s1 - L / 2)):
                    c0, c1 = _centered(mid, L, s0, s1)
                    c = Cut("rotation", round(c0, 2), round(c1, 2), round(r["score"] * (1 - 0.01 * k), 3))
                    if all(abs(c.start_s - o.start_s) > 0.05 for o in out if o.kind == "rotation"):
                        out.append(c)
        return out
    # orbit (gyro) and spin (pair turns while the camera holds)
    rot_at = None
    if gyro and gyro.get("gx"):
        fs = gyro["fs"]; t = gyro["t"]
        rot8 = net_rotation_deg(list(zip(gyro["gx"], gyro["gy"], gyro["gz"])), fs)
        seen = lambda a_, b_: (lambda v: sum(x["frac"] >= PAIR_SEEN for x in v) / len(v) if v else 0.0)(
            [x for x in fr if a_ <= x["t"] <= b_])
        best = None
        for i, deg in enumerate(rot8):
            if t[i] < lo or t[i] + ORBIT_WINDOW_S > hi or deg < ORBIT_MIN_DEG or seen(t[i], t[i] + ORBIT_WINDOW_S) < 0.75:
                continue
            if best is None or deg > best[1]:
                best = (t[i], deg)
        if best:
            c0, c1 = _centered(best[0] + ORBIT_WINDOW_S / 2, CLIP["orbit"], lo, hi)
            out.append(Cut("orbit", round(c0, 2), round(c1, 2), round(best[1], 1)))
        n6 = int(round(SPIN_WINDOW_S * fs))
        cum = [(0.0, 0.0, 0.0)]
        for gx, gy, gz in zip(gyro["gx"], gyro["gy"], gyro["gz"]):
            p = cum[-1]; cum.append((p[0] + gx / fs, p[1] + gy / fs, p[2] + gz / fs))

        def rot_at(t0):
            i = next((k for k, tt in enumerate(t) if tt >= t0), None)
            if i is None or i + n6 >= len(cum):
                return math.inf
            return math.degrees(math.sqrt(sum((cum[i + n6][k] - cum[i][k]) ** 2 for k in range(3))))
    if emb is not None and rot_at is not None:
        import numpy as np
        E = np.asarray(emb, float)
        norm = np.linalg.norm(E, axis=1); ok = norm > 0
        En = np.where(ok[:, None], E / np.where(ok, norm, 1)[:, None], 0)
        ts = [f["t"] for f in frames]
        best = None
        for i, t0 in enumerate(ts):
            if t0 < lo or t0 + SPIN_WINDOW_S > hi:
                continue
            idx = [k for k in range(i, len(ts)) if ts[k] <= t0 + SPIN_WINDOW_S]
            if len(idx) < 6 or not all(ok[k] and frames[k]["frac"] >= 0.05 for k in idx):
                continue
            sizes = [frames[k]["frac"] for k in idx]
            third = max(1, len(sizes) // 3)
            trend = (sum(sizes[-third:]) / third) / (sum(sizes[:third]) / third)
            if not 1 / SPIN_SIZE_RATIO <= trend <= SPIN_SIZE_RATIO:   # closing in / backing off
                continue
            drift = float(1 - min(En[idx] @ En[i]))
            if drift >= SPIN_DRIFT_MIN and rot_at(t0) < SPIN_GYRO_MAX and (best is None or drift > best[1]):
                best = (t0, drift)
        if best:
            c0, c1 = _centered(best[0] + SPIN_WINDOW_S / 2, CLIP["spin"], lo, hi)
            out.append(Cut("spin", round(c0, 2), round(c1, 2), round(best[1], 3)))
    return out


def plan(exit_s: float | None, drogue_s: float | None, deploy_s: float, frames: list[dict],
         gyro: dict | None = None, emb=None, length: float = EDIT_S,
         rotations: list[dict] | None = None, faces: list[tuple[float, float]] | None = None,
         order: str = "time", seed: int | str | None = None) -> list[Cut]:
    """The edit: exit clip, middle clips by WANT priority, deploy clip; ``length`` seconds
    in total, in time order, no overlaps."""
    if exit_s is None:
        head = None
    elif drogue_s is not None and drogue_s + EXIT_AFTER_DROGUE > exit_s + MIN_CLIP / 2:
        head = Cut("exit", round(max(0.0, exit_s - EXIT_PRE), 2), round(drogue_s + EXIT_AFTER_DROGUE, 2))
    else:
        head = Cut("exit", round(max(0.0, exit_s - EXIT_PRE), 2), round(exit_s + EXIT_POST, 2))
    tail = Cut("deploy", round(deploy_s - DEPLOY_PRE, 2), round(deploy_s + DEPLOY_POST, 2))
    lo = max(head.end_s if head else 0.0, drogue_s if drogue_s is not None else 0.0) + GAP
    hi = tail.start_s - GAP
    pool = candidates(frames, lo, hi, gyro, emb, rotations, faces)
    budget = length - tail.dur - (head.dur if head else 0.0)
    chosen: list[Cut] = []
    free = lambda c: all(c.end_s + GAP <= o.start_s or o.end_s + GAP <= c.start_s for o in chosen)
    for kind, n in WANT:
        for c in sorted((c for c in pool if c.kind == kind and c not in chosen), key=lambda c: -c.score):
            if n == 0 or sum(x.dur for x in chosen) + MIN_CLIP > budget:
                break
            if free(c):
                room = budget - sum(x.dur for x in chosen)
                if c.dur > room:
                    c = Cut(c.kind, c.start_s, round(c.start_s + room, 2), c.score)
                chosen.append(c); n -= 1
    # fill: any remaining candidate that fits, best first, trimmed to the room left
    for kind in FILL:
        for c in sorted((c for c in pool if c.kind == kind and c not in chosen), key=lambda c: -c.score):
            room = budget - sum(x.dur for x in chosen)
            if room < MIN_CLIP:
                break
            if free(c):
                chosen.append(c if c.dur <= room else Cut(c.kind, c.start_s, round(c.start_s + room, 2), c.score))
    # stretch: grow chosen clips into the free time around them, a little each
    orig = {id(c): c.dur for c in chosen}
    base = {}
    for _ in range(20):
        short = budget - sum(x.dur for x in chosen)
        if short < 0.05 or not chosen:
            break
        chosen.sort(key=lambda c: c.start_s)
        grew = False
        for i, c in enumerate(chosen):
            short = budget - sum(x.dur for x in chosen)
            if short < 0.05:
                break
            left = chosen[i - 1].end_s + GAP if i else lo
            right = chosen[i + 1].start_s - GAP if i + 1 < len(chosen) else hi
            step = min(0.5, short)
            d0 = base.get((c.kind, c.start_s, c.end_s), orig.get(id(c), c.dur))
            a = max(left, c.start_s - step / 2); b = min(right, a + c.dur + step)
            if b - a > c.dur + 0.01 and b - a - d0 <= MAX_STRETCH:
                n = Cut(c.kind, round(a, 2), round(b, 2), c.score)
                base[(n.kind, n.start_s, n.end_s)] = d0
                chosen[i] = n; grew = True
        if not grew:
            break
    # filler: free-fall pieces in the middle of the largest free gaps, when the
    # candidates cannot fill the time (e.g. the detector lost the pair)
    for _ in range(20):
        room = budget - sum(x.dur for x in chosen)
        if room < MIN_CLIP:
            break
        edges = [lo] + [v for c in sorted(chosen, key=lambda c: c.start_s) for v in (c.start_s - GAP, c.end_s + GAP)] + [hi]
        gaps = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] - edges[k] >= MIN_CLIP]
        if not gaps:
            break
        g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
        L = min(CLIP["medium"], room, g1 - g0)
        c0, c1 = _centered((g0 + g1) / 2, L, g0, g1)
        chosen.append(Cut("freefall", round(c0, 2), round(c1, 2)))
    short = round(budget - sum(x.dur for x in chosen), 2)
    if short > 0:                                    # last resort: lengthen deploy, then exit
        extra = min(short, DEPLOY_EXTRA_MAX)
        tail = Cut("deploy", tail.start_s, round(tail.end_s + extra, 2)); short = round(short - extra, 2)
        if short > 0 and head:
            head = Cut("exit", round(max(0.0, head.start_s - min(short, EXIT_EXTRA_MAX)), 2), head.end_s)
    middle = sorted(chosen, key=lambda c: c.start_s)
    if order == "random":
        import random
        random.Random(seed if seed is not None else round(deploy_s, 1)).shuffle(middle)
    return ([head] if head else []) + middle + [tail]


def face_options(frames: list[dict], lo: float, hi: float, top: int = 6) -> list[Cut]:
    """Every passenger-face clip of the free fall, best first — to pick from."""
    return sorted((c for c in candidates(frames, lo, hi) if c.kind == "face"), key=lambda c: -c.score)[:top]


def render(path: str, cuts: list[Cut], out: str, width: int = 1920, height: int = 1080, fps: int = 30,
           gpu: bool | None = None) -> bool:
    """Cut ``cuts`` from ``path`` and join them into ``out`` (H.264, no audio), each piece
    fitted inside width x height with its aspect kept (vertical footage gets side bars).
    On an NVIDIA GPU: NVDEC decode + scale_cuda, the camera's rotation applied after the
    download (NVDEC skips ffmpeg's autorotate), NVENC encode when the driver allows it
    (x264 otherwise); the all-CPU path is the fallback."""
    from tandem.video_io import gpu_decode_available, nvenc_available, video_size, rotation, _ROTATE
    use_gpu = gpu_decode_available() if gpu is None else gpu
    sw, sh = video_size(path)                         # display size, rotation applied
    k = min(width / sw, height / sh)
    fw, fh = int(sw * k) // 2 * 2, int(sh * k) // 2 * 2
    args = ["ffmpeg", "-v", "error", "-y"]
    for c in cuts:
        if use_gpu:
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        args += ["-ss", f"{c.start_s:.3f}", "-t", f"{c.dur:.3f}", "-i", path]
    if use_gpu:
        rot = rotation(path)
        pw, ph = (fh, fw) if rot in (90, 270) else (fw, fh)
        turn = f",{_ROTATE[rot]}" if rot in _ROTATE else ""
        scale = f"scale_cuda={pw}:{ph}:format=nv12,hwdownload,format=nv12{turn}"
    else:
        scale = f"scale={fw}:{fh}"
    chains = [f"[{i}:v]{scale},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}]"
              for i in range(len(cuts))]
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(cuts))) + f"concat=n={len(cuts)}:v=1:a=0[out]"
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"] if use_gpu and nvenc_available() else
           ["-c:v", "libx264", "-preset", "fast", "-crf", "20"])
    args += ["-filter_complex", graph, "-map", "[out]", *enc, "-pix_fmt", "yuv420p", out]
    ok = subprocess.run(args, capture_output=True).returncode == 0
    if not ok and use_gpu:
        return render(path, cuts, out, width, height, fps, gpu=False)
    return ok
