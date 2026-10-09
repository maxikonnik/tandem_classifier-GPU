import math

from tandem import freefall_edit as fe


def _frames(t0, t1, frac_at, face_at=lambda t: None, fps=2.0):
    out, t = [], t0
    while t <= t1:
        out.append({"t": round(t, 2), "frac": frac_at(t), "passenger": face_at(t)})
        t += 1 / fps
    return out


def _check_edit(cuts, length=fe.EDIT_S):
    assert abs(sum(c.dur for c in cuts) - length) < 0.05
    for a, b in zip(cuts, cuts[1:]):
        assert a.end_s <= b.start_s                                 # time order, no overlap


def test_plan_is_30_s_from_exit_to_deploy_in_time_order():
    exit_s, drogue_s, deploy_s = 50.0, 55.0, 100.0
    frac = lambda t: 0.06 if t < 70 else (0.25 if t < 85 else 0.6)
    face = lambda t: {"h": 0.09, "smile": 0.9, "jaw": 0.4} if 86 <= t <= 92 else None
    cuts = fe.plan(exit_s, drogue_s, deploy_s, _frames(drogue_s, deploy_s, frac, face))
    _check_edit(cuts)
    kinds = [c.kind for c in cuts]
    assert kinds[0] == "exit" and kinds[-1] == "deploy"
    assert cuts[0].start_s == exit_s - fe.EXIT_PRE and cuts[-1].start_s == deploy_s - fe.DEPLOY_PRE
    assert {"wide", "medium", "face"} <= set(kinds)


def test_plan_adds_an_orbit_from_the_gyro():
    exit_s, drogue_s, deploy_s = 10.0, 15.0, 60.0
    fs = 10.0
    t = [i / fs for i in range(700)]
    gx = [math.radians(45) if 300 <= i < 420 else 0.0 for i in range(700)]   # 30-42 s
    gyro = {"fs": fs, "t": t, "gx": gx, "gy": [0.0] * 700, "gz": [0.0] * 700}
    cuts = fe.plan(exit_s, drogue_s, deploy_s, _frames(drogue_s, deploy_s, lambda t: 0.2), gyro)
    _check_edit(cuts)
    orbit = [c for c in cuts if c.kind == "orbit"]
    assert len(orbit) == 1 and 28 <= orbit[0].start_s <= 38


def test_plan_without_exit_still_fills_the_length():
    cuts = fe.plan(None, None, 60.0, _frames(15.0, 60.0, lambda t: 0.2))
    _check_edit(cuts)
    assert cuts[-1].kind == "deploy"


def test_face_options_rank_by_emotion():
    face = lambda t: ({"h": 0.08, "smile": 0.95, "jaw": 0.1} if 20 <= t <= 23 else
                      {"h": 0.08, "smile": 0.2, "jaw": 0.1} if 30 <= t <= 33 else None)
    opts = fe.face_options(_frames(10, 50, lambda t: 0.5, face), 10, 50)
    assert len(opts) == 2 and 19 <= opts[0].start_s <= 23


def test_plan_takes_rotation_segments_from_the_detector():
    exit_s, drogue_s, deploy_s = 10.0, 15.0, 60.0
    rot = [{"start_s": 30.0, "end_s": 40.0, "score": 0.9}]
    cuts = fe.plan(exit_s, drogue_s, deploy_s, _frames(drogue_s, deploy_s, lambda t: 0.2), rotations=rot)
    _check_edit(cuts)
    r = [c for c in cuts if c.kind == "rotation"]
    assert len(r) == 1 and 30 <= r[0].start_s and r[0].end_s <= 40
    assert not any(c.kind in ("orbit", "spin") for c in cuts)       # the detector replaces the rules
