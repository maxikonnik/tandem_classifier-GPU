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


def _no_source_overlap(cuts):
    s = sorted(cuts, key=lambda c: c.start_s)
    return all(a.end_s <= b.start_s for a, b in zip(s, s[1:]))


def test_exit_clip_runs_to_half_a_second_after_the_drogue():
    cuts = fe.plan(50.0, 56.0, 100.0, _frames(56.0, 100.0, lambda t: 0.2))
    assert cuts[0].kind == "exit" and cuts[0].start_s == 49.0 and cuts[0].end_s == 56.5


def test_two_most_general_shots_are_always_taken():
    # the pair is small (3 %) only around 70-74 s and 88-92 s; elsewhere 25 %
    frac = lambda t: 0.03 if (70 <= t < 74 or 88 <= t < 92) else 0.25
    cuts = fe.plan(50.0, 55.0, 100.0, _frames(55.0, 100.0, frac))
    wide = [c for c in cuts if c.kind == "wide"]
    assert len(wide) >= 2
    assert all(69.5 <= c.start_s and c.end_s <= 74.5 or 87.5 <= c.start_s and c.end_s <= 92.5 for c in wide[:2])


def test_picked_faces_replace_the_automatic_ones():
    face = lambda t: {"h": 0.09, "smile": 0.95, "jaw": 0.4} if 60 <= t <= 64 else None
    cuts = fe.plan(50.0, 55.0, 100.0, _frames(55.0, 100.0, lambda t: 0.2, face), faces=[(80.0, 82.5)])
    f = [c for c in cuts if c.kind == "face"]
    assert len(f) == 1 and f[0].start_s == 80.0 and f[0].end_s == 82.5
    none = fe.plan(50.0, 55.0, 100.0, _frames(55.0, 100.0, lambda t: 0.2, face), faces=[])
    assert not any(c.kind == "face" for c in none)


def test_random_order_shuffles_the_middle_only_and_is_repeatable():
    fr = _frames(55.0, 100.0, lambda t: 0.03 if t < 65 else 0.2)
    a = fe.plan(50.0, 55.0, 100.0, fr, order="random", seed="jump")
    b = fe.plan(50.0, 55.0, 100.0, fr, order="random", seed="jump")
    t = fe.plan(50.0, 55.0, 100.0, fr)
    assert [c.as_dict() for c in a] == [c.as_dict() for c in b]
    assert a[0].kind == "exit" and a[-1].kind == "deploy"
    assert sorted(c.start_s for c in a) == sorted(c.start_s for c in t)
    assert [c.start_s for c in a[1:-1]] != [c.start_s for c in t[1:-1]]
    assert _no_source_overlap(a) and abs(sum(c.dur for c in a) - fe.EDIT_S) < 0.05


def test_picked_face_and_rotation_win_over_general_shots():
    # the widest shot (pair 3 %) sits exactly on the picked face and the rotation
    frac = lambda t: 0.03 if 60 <= t < 80 else 0.25
    cuts = fe.plan(50.0, 55.0, 100.0, _frames(55.0, 100.0, frac), faces=[(62.0, 64.5)],
                   rotations=[{"start_s": 66.0, "end_s": 78.0, "score": 0.9}])
    kinds = [c.kind for c in cuts]
    assert "face" in kinds and "rotation" in kinds and kinds.count("wide") >= 2
    assert _no_source_overlap(cuts)
