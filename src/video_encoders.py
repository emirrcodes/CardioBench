"""Echo-specific encoders from the CardioBench paper that are not open_clip models.

* ``panecho``   -- PanEcho video backbone (ConvNeXt-T frame encoder + its own temporal
  transformer), 16-frame clip -> 768-d; ImageNet-normalised input (CarDS-Yale/PanEcho, torch.hub).
* ``echoprime`` -- EchoPrime video encoder (torchvision MViT-v2-S, 512-d head), 16-frame clip;
  pixel values in 0-255 normalised with EchoPrime's own mean / std (echonet/EchoPrime v1.0.0).
* ``panecho_frames`` (via ``src.models``) -- PanEcho's ConvNeXt-T frame encoder alone, so the
  per-frame pipeline and the temporal heads can run on it.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

CLIP_LEN = 16
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
ECHOPRIME_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).view(3, 1, 1, 1)
ECHOPRIME_STD = torch.tensor([47.989223, 46.456997, 47.20083]).view(3, 1, 1, 1)
ECHOPRIME_ZIP = "https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip"
CACHE = Path(os.environ.get("CARDIOBENCH_CACHE", Path.home() / ".cache" / "cardiobench"))

VIDEO_ENCODERS = ("panecho", "echoprime")
CLIP_VIEWS = ("consecutive16", "stride2", "uniform16")


def clip_indices(n_frames: int, view: str) -> List[int]:
    """16 frame indices for a clip view (repeating the last frame when the clip is short).

    ``consecutive16``: frames 0-15. ``stride2``: every 2nd of the first 32 (EchoPrime's own
    sampling). ``uniform16``: 16 frames spread over the whole clip.
    """
    if n_frames <= 0:
        return []
    if view == "consecutive16":
        idx = list(range(min(CLIP_LEN, n_frames)))
    elif view == "stride2":
        idx = list(range(0, min(2 * CLIP_LEN, n_frames), 2))
    elif view == "uniform16":
        idx = [int(round(x)) for x in np.linspace(0, n_frames - 1, CLIP_LEN)]
    else:
        raise ValueError(f"Unknown clip view '{view}' (expected one of {CLIP_VIEWS})")
    return idx + [idx[-1]] * (CLIP_LEN - len(idx))


def _to_rgb_uint8(frames: np.ndarray, bgr: bool) -> np.ndarray:
    """(T, H, W[, C]) -> (T, H, W, 3) uint8 RGB."""
    f = np.asarray(frames)
    if f.ndim == 3:
        f = f[..., None]
    if f.shape[-1] == 1:
        f = np.repeat(f, 3, axis=-1)
    elif bgr:
        f = f[..., ::-1]
    return np.ascontiguousarray(f[..., :3]).astype(np.uint8)


def _hub_load_panecho(**kw):
    """torch.hub.load for PanEcho, whose hubconf imports its own top-level ``src`` package.

    This repo also has ``src`` (a regular package, which wins over PanEcho's namespace ``src`` on
    sys.path): hide ours from ``sys.modules`` and the repo root from ``sys.path`` while PanEcho
    loads, then restore both.
    """
    repo_root = Path(__file__).resolve().parents[1]
    saved_path = list(sys.path)
    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != repo_root]
    try:
        return torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", trust_repo=True, **kw)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(ours)


def _echoprime_weights() -> Path:
    target = CACHE / "echoprime" / "echo_prime_encoder.pt"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    zpath = target.parent / "model_data.zip"
    if not zpath.exists():
        print(f"[video_encoders] downloading {ECHOPRIME_ZIP} (~1.3 GB)", flush=True)
        urllib.request.urlretrieve(ECHOPRIME_ZIP, zpath)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith("echo_prime_encoder.pt"))
        with z.open(name) as src, open(target, "wb") as dst:
            dst.write(src.read())
    zpath.unlink()  # only the encoder is needed
    return target


def load_video_encoder(name: str, device: str, pretrained: bool = True
                       ) -> Tuple[torch.nn.Module, Callable[[np.ndarray, bool], torch.Tensor]]:
    """Return ``(model, prep)``; ``prep(frames, bgr)`` maps 16 frames (T, 224, 224[, C]) uint8
    to the model's (3, 16, 224, 224) float input."""
    if name == "panecho":
        model = _hub_load_panecho(backbone_only=True, clip_len=CLIP_LEN, pretrained=pretrained)

        def prep(frames, bgr):
            x = torch.from_numpy(_to_rgb_uint8(frames, bgr)).permute(3, 0, 1, 2).float() / 255.0
            return (x - IMAGENET_MEAN) / IMAGENET_STD
    elif name == "echoprime":
        import torchvision

        model = torchvision.models.video.mvit_v2_s()
        model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 512)
        if pretrained:
            model.load_state_dict(torch.load(_echoprime_weights(), map_location="cpu"))

        def prep(frames, bgr):
            x = torch.from_numpy(_to_rgb_uint8(frames, bgr)).permute(3, 0, 1, 2).float()
            return (x - ECHOPRIME_MEAN) / ECHOPRIME_STD
    else:
        raise ValueError(f"Unknown video encoder '{name}' (expected one of {VIDEO_ENCODERS})")
    return model.eval().to(device), prep


class PanEchoFrames(torch.nn.Module):
    """PanEcho's ConvNeXt-T frame encoder behind the ``encode_image`` interface of open_clip."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = _hub_load_panecho(image_encoder_only=True, pretrained=pretrained)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def panecho_frame_transform():
    import torchvision.transforms as T

    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN.flatten().tolist(), IMAGENET_STD.flatten().tolist()),
    ])
