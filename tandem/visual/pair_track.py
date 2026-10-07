"""The tandem pair through a span of video, on the GPU.

For frames sampled at ``fps`` (decoded by tandem.video_io — NVDEC, rotation applied):
- people: RT-DETR (``PekingU/rtdetr_r50vd``, COCO "person") in fp16 on the GPU;
- the pair: the largest group of overlapping person boxes — other jumpers are separate
  groups; a small box cut by two frame edges is the operator's own arm and is dropped;
- embeddings (DINOv2, tandem.visual.probe) of the pair crop and of the whole frame:
  how the pair's look and the scene change through the span.
"""
from __future__ import annotations

import numpy as np

FRAME_W = 960
_det = None


def _detector():
    """RT-DETR on the GPU in fp16 (CPU fp32 without CUDA). Loaded once per process."""
    global _det
    if _det is None:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        name = "PekingU/rtdetr_r50vd"
        try:
            proc = RTDetrImageProcessor.from_pretrained(name, local_files_only=True)
            model = RTDetrForObjectDetection.from_pretrained(name, local_files_only=True)
        except OSError:
            proc = RTDetrImageProcessor.from_pretrained(name)
            model = RTDetrForObjectDetection.from_pretrained(name)
        cuda = torch.cuda.is_available()
        model = (model.half().cuda() if cuda else model).eval()
        person = [k for k, v in model.config.id2label.items() if v == "person"][0]
        _det = (proc, model, person, cuda)
    return _det


def person_boxes(frames, threshold: float = 0.5) -> list[list[list[float]]]:
    """Person boxes [x0, y0, x1, y1] in pixels for each uint8 RGB frame [N, H, W, 3]."""
    import torch
    import torch.nn.functional as F
    proc, model, person, cuda = _detector()
    H, W = frames.shape[1:3]
    dev, dtype = ("cuda", torch.float16) if cuda else ("cpu", torch.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(frames), 16):
            x = torch.from_numpy(np.ascontiguousarray(frames[s:s + 16])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False, antialias=True).to(dtype)
            o = model(pixel_values=x)
            o.logits, o.pred_boxes = o.logits.float(), o.pred_boxes.float()
            res = proc.post_process_object_detection(o, threshold=threshold, target_sizes=[(H, W)] * x.shape[0])
            out += [[b.tolist() for b, l in zip(r["boxes"], r["labels"]) if int(l) == person] for r in res]
    return out


def pair_box(boxes, W, H):
    """The pair: the largest group of overlapping person boxes -> (box or None, share of frame)."""
    def edge_hand(b):
        touches = (b[0] <= 2) + (b[1] <= 2) + (b[2] >= W - 2) + (b[3] >= H - 2)
        return touches >= 2 and (b[2] - b[0]) * (b[3] - b[1]) < 0.25 * W * H
    groups = []
    for b in (b for b in boxes if not edge_hand(b)):
        hit = [g for g in groups if any(not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]) for o in g)]
        groups = [g for g in groups if g not in hit] + [[b] + [o for g in hit for o in g]]
    if not groups:
        return None, 0.0
    ub = lambda g: [max(0, min(o[0] for o in g)), max(0, min(o[1] for o in g)),
                    min(W, max(o[2] for o in g)), min(H, max(o[3] for o in g))]
    best = max((ub(g) for g in groups), key=lambda u: (u[2] - u[0]) * (u[3] - u[1]))
    return best, (best[2] - best[0]) * (best[3] - best[1]) / (W * H)


def track(path: str, lo: float, hi: float, fps: float = 2.0) -> dict:
    """{t, frac, box (normalised or None), crop_emb [N, 384] (zeros without a pair),
    frame_emb [N, 384]} for frames at ``fps`` in [lo, hi)."""
    from PIL import Image
    from tandem.video_io import read_frames, fit_width
    from tandem.visual import probe
    W, H = fit_width(path, FRAME_W)
    ts, frames = read_frames(path, lo, hi, W, H, fps=fps)
    empty = np.zeros((0, 384), np.float32)
    if len(frames) == 0:
        return {"t": np.zeros(0), "frac": np.zeros(0), "box": [], "crop_emb": empty, "frame_emb": empty}
    _, model = probe._load()
    boxes = person_boxes(frames)
    fracs, nbox, crops, have = [], [], [], []
    for f, bx in zip(frames, boxes):
        box, frac = pair_box(bx, W, H)
        fracs.append(frac)
        if box and frac >= 0.005:
            x0, y0, x1, y1 = (int(v) for v in box)
            crops.append(np.asarray(Image.fromarray(f[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]).resize((224, 224))))
            have.append(True)
            nbox.append([box[0] / W, box[1] / H, box[2] / W, box[3] / H])
        else:
            have.append(False); nbox.append(None)
    crop_emb = np.zeros((len(frames), 384), np.float32)
    if crops:
        crop_emb[np.array(have)] = probe.embed_arrays(np.stack(crops), model)
    frame_emb = probe.embed_arrays(np.stack([np.asarray(Image.fromarray(f).resize((224, 224))) for f in frames]), model)
    return {"t": np.asarray(ts, float), "frac": np.asarray(fracs, float), "box": nbox,
            "crop_emb": crop_emb, "frame_emb": frame_emb}
