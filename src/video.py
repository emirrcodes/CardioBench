from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def crop_and_scale(img: np.ndarray, res: Tuple[int, int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.1) -> np.ndarray:
    """
    Preprocess an image by center-cropping and resizing to the target resolution.
    """
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding] if padding > 0 else img
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding] if padding > 0 else img
    if zoom > 0:
        pad_x = int(round(img.shape[1] * zoom))
        pad_y = int(round(img.shape[0] * zoom))
        if pad_x > 0 and pad_y > 0 and pad_y * 2 < img.shape[0] and pad_x * 2 < img.shape[1]:
            img = img[pad_y:-pad_y, pad_x:-pad_x]

    return cv2.resize(img, res, interpolation=interpolation)


def read_avi(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Read echocardiography videos in AVI format and retunr as a numpy array of shape (T, H, W, C).
    """
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to the range [0, 255] as uint8.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        mid = arr.shape[-1] // 2
        arr = arr[..., mid]
    if arr.ndim == 1:
        arr = arr[:, None]
    smin = float(arr.min()) if arr.size else 0.0
    smax = float(arr.max()) if arr.size else 0.0
    if smax > smin:
        arr = (arr - smin) / (smax - smin)
    else:
        arr = arr * 0.0
    arr = (arr * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def _nifti_layout(shape: Sequence[int]) -> Tuple[int, bool]:
    """Return ``(n_frames, is_rgb)`` for a NIfTI cine.

    Most volumes are ``(H, W, T)``; some CardiacNet controls (the ``PHI*`` files) are
    ``(H, W, T, 3)`` RGB. Treating the last axis as time there yields 3 "frames" that are
    the colour channels of a single mid-clip image, and since those files are all
    negatives the probe can learn the file format instead of the pathology.
    """
    if len(shape) == 4 and shape[-1] in (3, 4):
        return int(shape[2]), True
    if len(shape) == 2:
        return 1, False
    return int(shape[-1]), False


def _nifti_frame(dataobj, shape: Sequence[int], idx: int, is_rgb: bool) -> np.ndarray:
    if len(shape) == 2:
        return _normalize_slice(np.asarray(dataobj))
    if is_rgb:
        rgb = np.asarray(dataobj[:, :, idx, :3], dtype=np.float32)
        # Luminance, so these clips look like the grayscale cines of every other video.
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return _normalize_slice(gray)
    return _normalize_slice(np.asarray(dataobj[..., idx]))


def read_nii(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Read echocardiography videos in NIfTI format and return as a numpy array of shape (T, H, W, C).
    """
    img = nib.load(str(path))
    n_frames, is_rgb = _nifti_layout(img.shape)
    frames: List[np.ndarray] = []
    for idx in range(n_frames):
        frame = _nifti_frame(img.dataobj, img.shape, idx, is_rgb)
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    return np.stack(frames, axis=0)


def read_video(path: Path, res: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Main video reading function that supports NIfTI, AVI, and image files.
    """
    name = path.name.lower()
    suf = path.suffix.lower()
    if name.endswith(".nii.gz") or suf == ".nii":
        return read_nii(path, res=res)
    if suf in {".avi", ".mp4"}:
        return read_avi(path, res=res)
    if suf in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if res is not None:
            img = crop_and_scale(img, res)
        if img.ndim == 2:
            frames = np.expand_dims(img, axis=0)
        else:
            frames = np.expand_dims(img, axis=0)
        return frames
    raise ValueError(f"Unsupported video type for {path}")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii.gz") or path.suffix.lower() == ".nii"


def select_frame_indices(
    n_frames: int, key_frame: int, max_frames: int, stride: int, sampling: str = "consecutive"
) -> List[int]:
    """``consecutive``: ``max_frames`` frames from ``key_frame`` on (the original behaviour,
    ~0.5 s of a 30-50 fps cine). ``uniform``: ``max_frames`` frames spread evenly over the
    whole clip, so the embedding sees every phase of the cardiac cycle. ``both``: the union
    of the two, so one embedding pass serves either view (see ``frame_view_positions``)."""
    if sampling == "consecutive":
        return indices_after_keyframe(n_frames, key_frame, max_frames, stride)
    if sampling == "uniform":
        if n_frames <= 0:
            return []
        take = min(max_frames, n_frames)
        return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, take)})
    if sampling == "both":
        return sorted(
            set(select_frame_indices(n_frames, key_frame, max_frames, stride, "consecutive"))
            | set(select_frame_indices(n_frames, key_frame, max_frames, stride, "uniform"))
        )
    raise ValueError(f"Unknown sampling '{sampling}' (expected consecutive|uniform|both)")


def frame_view_positions(
    frame_indices: Sequence[int], n_frames: int, view: str, *, stored_max: int
) -> List[int]:
    """Positions into ``frame_indices`` (frames embedded with ``sampling="both"`` and
    ``max_frames=stored_max``) that make up ``view``, e.g. ``consecutive16`` or ``uniform8``.

    ``consecutiveK``: the first K frames of the clip (K <= stored_max). ``uniformK``: every
    (stored_max / K)-th frame of the uniform set, so K must divide ``stored_max``.
    ``all``: every stored frame.
    """
    pos = {int(f): i for i, f in enumerate(frame_indices)}
    if view == "all":
        return list(range(len(frame_indices)))
    for kind in ("consecutive", "uniform"):
        if view.startswith(kind):
            k = int(view[len(kind):])
            if k > stored_max or (kind == "uniform" and stored_max % k):
                raise ValueError(f"view {view} needs K <= {stored_max} (and dividing it for uniform)")
            target = select_frame_indices(n_frames, 0, stored_max, 1, kind)
            target = target[:k] if kind == "consecutive" else target[:: stored_max // k]
            return [pos[f] for f in target if f in pos]
    raise ValueError(f"Unknown frame view '{view}'")


CYCLE_VIEWS_HELP = "edes | halfcycleN | edN"


def is_cycle_view(view: str) -> bool:
    return view == "edes" or view.startswith("halfcycle") or (view.startswith("ed") and view[2:].isdigit())


def cycle_view_indices(n_frames: int, ed: int, es: int, view: str) -> List[int]:
    """Frames anchored on the traced end-diastole (``ed``) and end-systole (``es``) frames.

    ``edes``: just those two frames. ``halfcycleN``: N frames spread from one to the other
    (systole or diastole, whichever the clip traced). ``edN``: N consecutive frames from ED.
    """
    last = max(n_frames - 1, 0)
    ed, es = min(max(int(ed), 0), last), min(max(int(es), 0), last)
    if view == "edes":
        return sorted({ed, es})
    if view.startswith("halfcycle"):
        a, b = sorted((ed, es))
        return sorted({int(round(x)) for x in np.linspace(a, b, int(view[len("halfcycle"):]))})
    if view.startswith("ed") and view[2:].isdigit():
        return list(range(ed, min(ed + int(view[2:]), n_frames)))
    raise ValueError(f"Unknown cycle view '{view}' (expected {CYCLE_VIEWS_HELP})")


def cycle_frame_indices(n_frames: int, ed: int, es: int, k: int = 16) -> List[int]:
    """Union of ``edes``, ``halfcycle{k}`` and ``ed{k}``: what ``--sampling cycle`` embeds."""
    out = set()
    for view in ("edes", f"halfcycle{k}", f"ed{k}"):
        out.update(cycle_view_indices(n_frames, ed, es, view))
    return sorted(out)


def read_clip(
    path: Path,
    *,
    res: Optional[Tuple[int, int]] = None,
    key_frame: int = 0,
    max_frames: int = 16,
    stride: int = 1,
    sampling: str = "consecutive",
    selector=None,
) -> Tuple[np.ndarray, int, List[int]]:
    """
    Read only the frames selected by ``select_frame_indices``.

    Returns ``(frames, n_frames_raw, selected_indices)``. NIfTI volumes are memory-mapped,
    so only the selected slices are decoded instead of the whole cine (CardiacNet
    volumes are ~100 MB each). Other formats fall back to a full read. ``selector``, if
    given, maps the clip's frame count to explicit frame indices and overrides ``sampling``.
    """
    def pick(n):
        return selector(n) if selector is not None else select_frame_indices(n, key_frame, max_frames, stride, sampling)

    if _is_nifti(path):
        img = nib.load(str(path))
        n_raw, is_rgb = _nifti_layout(img.shape)
        sel = pick(n_raw)
        frames: List[np.ndarray] = []
        for idx in sel:
            frame = _nifti_frame(img.dataobj, img.shape, idx, is_rgb)
            if res is not None:
                frame = crop_and_scale(frame, res)
            frames.append(frame)
        if not frames:
            return np.empty((0,)), n_raw, sel
        return np.stack(frames, axis=0), n_raw, sel

    frames_all = read_video(path, res=res)
    n_raw = int(frames_all.shape[0])
    sel = pick(n_raw)
    return frames_all[sel], n_raw, sel


def preprocess_frames(frames: np.ndarray, preprocess_val) -> torch.Tensor:
    to_pil = T.ToPILImage()
    tensors = [preprocess_val(to_pil(frame)) for frame in frames]
    return torch.stack(tensors, dim=0)


@torch.inference_mode()
def encode_video_clip_batched(
    model,
    frames_tensor: torch.Tensor,
    *,
    device: str = "cuda",
    precision: str = "bf16",
    batch_size: int = 128,
    use_channels_last: bool = True,
    pin_memory: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    if pin_memory and frames_tensor.device.type == "cpu":
        frames_tensor = frames_tensor.pin_memory()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    outputs: List[torch.Tensor] = []
    stream = torch.cuda.Stream(device=device) if "cuda" in device else None

    for start in range(0, frames_tensor.shape[0], batch_size):
        chunk = frames_tensor[start:start + batch_size]
        if stream is not None:
            with torch.cuda.stream(stream):
                x = chunk.to(device, dtype=dtype, non_blocking=True)
                if use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                x = model.encode_image(x)
                if normalize:
                    x = F.normalize(x, dim=-1)
        else:
            x = chunk.to(device, dtype=dtype)
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            x = model.encode_image(x)
            if normalize:
                x = F.normalize(x, dim=-1)
        outputs.append(x)

    if stream is not None:
        torch.cuda.current_stream(device).wait_stream(stream)

    feats = torch.cat(outputs, dim=0).to("cpu").to(torch.float16)
    return feats


def indices_after_keyframe(n_frames: int, start: int, max_frames: int, stride: int) -> List[int]:
    if n_frames <= 0:
        return []
    start = max(0, min(start, n_frames - 1))
    idxs = list(range(start, n_frames, max(1, stride)))
    return idxs[:max_frames]


def select_indices(n_frames: int, max_frames: int, stride: int) -> List[int]:
    end = min(max_frames, n_frames)
    return list(range(0, end, max(1, stride)))


def save_sample_frames(
    frames: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    indices: Optional[Sequence[int]] = None,
    max_frames: int = 3,
) -> List[Path]:
    """
    Save sample frames from a video to the specified output directory for visual inspection
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if indices is None:
        take = min(max_frames, int(frames.shape[0]))
        indices = list(range(take))

    saved: List[Path] = []
    for idx in indices:
        if idx < 0 or idx >= frames.shape[0]:
            continue
        frame = np.squeeze(frames[idx])
        if frame.dtype != np.uint8:
            arr = frame.astype(np.float32)
            arr -= arr.min() if arr.size else 0.0
            max_val = arr.max() if arr.size else 0.0
            if max_val > 0:
                arr /= max_val
            frame_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_u8 = frame
        if frame_u8.ndim == 3 and frame_u8.shape[-1] == 1:
            frame_u8 = frame_u8[..., 0]
        out_path = out_dir / f"{stem}_frame{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_u8)
        saved.append(out_path)
    return saved


__all__ = [
    "crop_and_scale",
    "read_video",
    "read_clip",
    "select_frame_indices",
    "frame_view_positions",
    "cycle_view_indices",
    "cycle_frame_indices",
    "is_cycle_view",
    "preprocess_frames",
    "encode_video_clip_batched",
    "indices_after_keyframe",
    "select_indices",
    "save_sample_frames",
]
