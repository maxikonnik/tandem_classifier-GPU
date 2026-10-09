"""Per-frame analysis of the free fall on the GPU: the pair, the passenger's face.

For frames sampled at ``fps`` in [lo, hi] (decoded by tandem.video_io — NVDEC, camera
rotation applied):
- the tandem pair: RT-DETR person boxes on the GPU (tandem.visual.pair_track); the pair
  is the largest group of overlapping boxes. RT-DETR keeps the pair where MediaPipe's
  EfficientDet loses it (dark suits against bright cloud, shot from below);
- faces: MediaPipe Face Landmarker on four overlapping, upscaled tiles of the pair box
  (faces in free fall are small; tiling finds them in 26 of 36 close-up frames vs 17 on
  the whole box). Each face: centre, height as a share of the frame height, smile
  (mean of mouthSmileLeft/Right), jawOpen (scream / laugh) and head yaw. The passenger
  is the lowest face — the instructor is behind and above them;
- with ``embed``: DINOv2 embeddings of the pair crop (``emb``) and of the whole frame
  (``femb``) — the inputs of the rotation detector (tandem.rotation).
Model files of MediaPipe live in tandem/visual/models/ and are fetched on first use.
"""
from __future__ import annotations

import math
import os
import urllib.request

import numpy as np

from tandem.visual.pair_track import pair_box, person_boxes  # noqa: F401  (pair_box re-exported)

MODELS = os.path.join(os.path.dirname(__file__), "models")
URLS = {"face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                                "face_landmarker/float16/latest/face_landmarker.task"}
FRAME_W = 1280          # analysis width; faces need the resolution
FACE_MIN_PAIR = 0.08    # look for faces only when the pair fills >= 8 % of the frame
_faces = None


def _model(name: str) -> str:
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        os.makedirs(MODELS, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def _face_landmarker():
    global _faces
    if _faces is None:
        from mediapipe.tasks.python import BaseOptions, vision
        _faces = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model("face_landmarker.task")),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            num_faces=3, min_face_detection_confidence=0.3))
    return _faces


def _mp_image(img):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img.convert("RGB"))))


def faces_in(img, box) -> list[dict]:
    """Faces inside the pair box, found on 2x2 overlapping upscaled tiles."""
    fl = _face_landmarker()
    W, H = img.size
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    found: list[dict] = []
    for i in (0, 1):
        for j in (0, 1):
            c = (x0 + i * w / 2 - 0.1 * w * (i > 0), y0 + j * h / 2 - 0.1 * h * (j > 0),
                 x0 + (i + 1) * w / 2 + 0.1 * w * (i < 1), y0 + (j + 1) * h / 2 + 0.1 * h * (j < 1))
            tile = img.crop(tuple(int(v) for v in c))
            s = 640 / max(tile.size)
            tile = tile.resize((max(1, int(tile.size[0] * s)), max(1, int(tile.size[1] * s))))
            r = fl.detect(_mp_image(tile))
            for lm, bs, mat in zip(r.face_landmarks, r.face_blendshapes, r.facial_transformation_matrixes):
                ys = [p.y for p in lm]; xs = [p.x for p in lm]
                cx = c[0] + np.mean(xs) * (c[2] - c[0]); cy = c[1] + np.mean(ys) * (c[3] - c[1])
                fh = (max(ys) - min(ys)) * (c[3] - c[1]) / H
                if any(abs(cx - f["cx"] * W) < 0.08 * w and abs(cy - f["cy"] * H) < 0.08 * h for f in found):
                    continue
                b = {k.category_name: k.score for k in bs}
                R = np.asarray(mat)[:3, :3]
                found.append({"cx": float(cx / W), "cy": float(cy / H), "h": float(fh),
                              "smile": float((b["mouthSmileLeft"] + b["mouthSmileRight"]) / 2),
                              "jaw": float(b["jawOpen"]),
                              "yaw": float(math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))))})
    return found


def analyze(path: str, lo: float, hi: float, fps: float = 2.0, embed: bool = False) -> list[dict]:
    """One dict per sampled frame: t, frac (pair share), box (normalised or None), faces,
    passenger (the lowest face or None) and, with ``embed``, emb (pair crop) and femb
    (whole frame) DINOv2 embeddings."""
    from PIL import Image
    from tandem.video_io import read_frames, fit_width
    W, H = fit_width(path, FRAME_W)
    ts, frames = read_frames(path, lo, hi, W, H, fps=fps)
    if len(frames) == 0:
        return []
    out: list[dict] = []
    crops = []
    for t, f, boxes in zip(ts, frames, person_boxes(frames)):
        box, frac = pair_box(boxes, W, H)
        img = Image.fromarray(f) if box and frac >= FACE_MIN_PAIR else None
        faces = faces_in(img, box) if img is not None else []
        out.append({"t": round(float(t), 3), "frac": round(frac, 4),
                    "box": [round(box[0] / W, 4), round(box[1] / H, 4), round(box[2] / W, 4), round(box[3] / H, 4)]
                    if box else None,
                    "faces": faces, "passenger": max(faces, key=lambda q: q["cy"]) if faces else None})
        if embed and box and frac >= 0.005:
            x0, y0, x1, y1 = (int(v) for v in box)
            crops.append((len(out) - 1, np.asarray(Image.fromarray(
                f[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]).resize((224, 224)))))
    if embed:
        from tandem.visual import probe
        _, model = probe._load()
        if crops:
            for (i, _), e in zip(crops, probe.embed_arrays(np.stack([c for _, c in crops]), model)):
                out[i]["emb"] = e
        fem = probe.embed_arrays(np.stack([np.asarray(Image.fromarray(f).resize((224, 224))) for f in frames]), model)
        for rec, e in zip(out, fem):
            rec["femb"] = e
    return out


def rotation_inputs(frames: list[dict]):
    """(t, frac, crop_emb, frame_emb, box) arrays for tandem.rotation from analyze(embed=True)."""
    z = np.zeros(384, np.float32)
    return (np.array([f["t"] for f in frames], float), np.array([f["frac"] for f in frames], float),
            np.stack([f.get("emb", z) for f in frames]) if frames else np.zeros((0, 384)),
            np.stack([f.get("femb", z) for f in frames]) if frames else np.zeros((0, 384)),
            [f["box"] for f in frames])
