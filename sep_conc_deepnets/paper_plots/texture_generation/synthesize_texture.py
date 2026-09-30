"""Standalone texture synthesis by matching Kymatio Scattering2D coefficients.

Same load / L-BFGS / histogram-match loop as wavelet-texture-synthesis, with
Kymatio Scattering2D in place of that project's rectifier-wavelet covariances.
Requirements: Kymatio from https://github.com/kymatio/kymatio
To reproduce the results in the paper, run:
    python synthesize_texture.py --image 14.jpg --J 4 --L 8
    python synthesize_texture.py --image 20.jpg --J 4 --L 8
    python plot_generation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.interpolate
import torch
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from PIL import Image


def load_image(path, size, gray=False, device="cpu"):
    """Load an image as float32 [0, 1], shape (1, C, size, size)."""
    im = Image.open(path)
    if gray or im.mode in ("L", "I", "F"):
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        if arr.shape != (size, size):
            arr = np.array(
                Image.fromarray((arr * 255).astype(np.uint8)).resize(
                    (size, size), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        tensor = torch.from_numpy(arr)[None, None]
    else:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        if arr.shape[0] != size or arr.shape[1] != size:
            arr = np.array(
                Image.fromarray((arr * 255).astype(np.uint8)).resize(
                    (size, size), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def scattering_mean(scattering, x):
    """Spatial mean of scattering coefficients: (B, C, D) or (B, D)."""
    s = scattering(x.contiguous())
    return s.mean(dim=(-2, -1))


def color_corr(im):
    """3x3 channel correlation of a (1, 3, H, W) image, as in routine_color.py."""
    x = im.view(3, -1)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-8)
    return (x @ x.T) / x.shape[-1]


def uniform_hist(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks / len(x)


def _match_channel(org, match, n_bins=100):
    hist, bin_edges = np.histogram(match.ravel(), bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
    inv_cdf = scipy.interpolate.interp1d(
        cum_values, bin_edges, bounds_error=False,
        fill_value=(bin_edges[0], bin_edges[-1]),
    )
    r = uniform_hist(org.ravel())
    r = np.clip(r, cum_values.min(), cum_values.max())
    return inv_cdf(r).reshape(org.shape)


def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    if grey or org_image.ndim == 2:
        return _match_channel(org_image, match_image, n_bins=n_bins)
    matched = np.zeros_like(org_image)
    for i in range(org_image.shape[-1]):
        matched[..., i] = _match_channel(
            org_image[..., i], match_image[..., i], n_bins=n_bins
        )
    return matched


SNAPSHOT_EVERY = 100


def snapshot_canvas(x, ref, gray, no_hist, out):
    """Write the current canvas to `out`, overwriting any previous file."""
    im_opt = to_hwc(x)
    im_ref = to_hwc(ref)
    if not no_hist:
        im_opt = histogram_matching(im_opt, im_ref, grey=gray)
    save_png(im_opt, out)


def synthesize(ref, scattering, maxite, seed, on_snapshot=None, snapshot_every=SNAPSHOT_EVERY):
    """Optimize a canvas so its scattering means (and RGB color corr) match `ref`."""
    torch.manual_seed(seed)
    mean = ref.mean(dim=(-2, -1), keepdim=True)
    std = ref.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    x = torch.normal(mean.expand_as(ref), std.expand_as(ref))
    x.requires_grad_(True)

    with torch.no_grad():
        s_ref = scattering_mean(scattering, ref)
        c_ref = color_corr(ref) if ref.shape[1] == 3 else None

    chunk = min(snapshot_every, maxite)
    optimizer = torch.optim.LBFGS(
        [x],
        max_iter=chunk,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        history_size=50,
    )

    n_calls = [0]

    def closure():
        optimizer.zero_grad()
        s_x = scattering_mean(scattering, x)
        loss = F.mse_loss(s_x, s_ref)
        if c_ref is not None:
            loss = loss + ((color_corr(x) - c_ref) ** 2).sum()
        loss.backward()
        n_calls[0] += 1
        if n_calls[0] == 1 or n_calls[0] % 20 == 0:
            print(f"  iter {n_calls[0]:4d}  loss={loss.item():.6e}")
        return loss

    print(f"Optimizing ({maxite} L-BFGS iterations, snapshot every {snapshot_every})...")
    done = 0
    while done < maxite:
        optimizer.max_iter = min(snapshot_every, maxite - done)
        optimizer.step(closure)
        done += optimizer.max_iter
        if on_snapshot is not None:
            on_snapshot(x.detach(), done)
    print(f"  done after {n_calls[0]} closure calls")
    return x.detach()


def to_hwc(tensor):
    arr = tensor.detach().cpu().squeeze(0).numpy()
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def save_png(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        im = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    else:
        im = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Synthesize a texture by matching Kymatio Scattering2D coefficients."
    )
    p.add_argument("--image", required=True, type=str,
                   help="Path to the example texture, e.g. 1.png")
    p.add_argument("--size", type=int, default=256,
                   help="Square resize (must be divisible by 2**J)")
    p.add_argument("--J", type=int, default=6, help="Scattering scale")
    p.add_argument("--L", type=int, default=6, help="Number of angles")
    p.add_argument("--maxite", type=int, default=500, help="L-BFGS iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG (default: <stem>_synth.png)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--gray", action="store_true", help="Force grayscale")
    p.add_argument("--no-hist", action="store_true",
                   help="Skip histogram matching")
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Example image not found: {image_path}")
    if args.size % (2 ** args.J) != 0:
        raise ValueError(
            f"--size {args.size} must be divisible by 2**J = {2 ** args.J}"
        )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    ref = load_image(image_path, args.size, gray=args.gray, device=device)
    gray = ref.shape[1] == 1
    print(f"Reference: {image_path}  shape={tuple(ref.shape)}")

    scattering = Scattering2D(J=args.J, shape=(args.size, args.size), L=args.L)
    scattering = scattering.to(device)
    scattering.eval()
    for p in scattering.parameters():
        p.requires_grad_(False)

    out = Path(args.out) if args.out else Path(f"{image_path.stem}_synth.png")

    def on_snapshot(x, n_iter):
        snapshot_canvas(x, ref, gray, args.no_hist, out)
        print(f"  saved {out} at L-BFGS iter {n_iter}")

    synthesize(ref, scattering, args.maxite, args.seed, on_snapshot=on_snapshot)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
