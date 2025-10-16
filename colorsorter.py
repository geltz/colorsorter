#!/usr/bin/env python3
import argparse
import math
import os
import logging
from pathlib import Path
import shutil
from typing import List, Tuple, Optional

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e

try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from e


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def is_in_dir(child: Path, base: Path) -> bool:
    try:
        child.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def list_images(folder: Path, recursive: bool, exclude_dir: Optional[Path], follow_symlinks: bool) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    out: List[Path] = []
    for p in it:
        try:
            if not p.is_file():
                continue
            if not follow_symlinks and p.is_symlink():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if exclude_dir is not None and (exclude_dir == p or is_in_dir(p, exclude_dir)):
                continue
            out.append(p)
        except PermissionError:
            logging.warning("Skipping (no permission): %s", p)
        except FileNotFoundError:
            logging.warning("Skipping (vanished): %s", p)
    return sorted(out)


def load_rgba(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        try:
            img.draft("RGB", (max_side, max_side))
        except Exception:
            pass
        img = img.convert("RGBA")
        w, h = img.size
        if max(w, h) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        arr = np.asarray(img)
    return arr.astype(np.float32) / 255.0  # HxWx4 in [0,1]


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc <= 1e-7, 0.0, delta / (maxc + 1e-7))
    h = np.zeros_like(maxc)

    mask = delta > 1e-7
    rc = np.where(mask, (g - b) / (delta + 1e-7), 0.0)
    gc = np.where(mask, (b - r) / (delta + 1e-7), 0.0)
    bc = np.where(mask, (r - g) / (delta + 1e-7), 0.0)

    r_is_max = (maxc == r) & mask
    g_is_max = (maxc == g) & mask
    b_is_max = (maxc == b) & mask

    h = np.where(r_is_max, (rc % 6.0), h)
    h = np.where(g_is_max, (gc + 2.0), h)
    h = np.where(b_is_max, (bc + 4.0), h)

    h = (h * 60.0) % 360.0
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> Optional[float]:
    wsum = float(np.sum(weights))
    if wsum <= 1e-8:
        return None
    ang = np.deg2rad(angles_deg)
    x = float(np.sum(np.cos(ang) * weights))
    y = float(np.sum(np.sin(ang) * weights))
    if abs(x) <= 1e-8 and abs(y) <= 1e-8:
        return None
    mean = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return mean


def image_hue_score(path: Path, max_side: int, sat_thresh: float, val_thresh: float, alpha_thresh: float) -> Tuple[bool, Optional[float], float]:
    arr = load_rgba(path, max_side=max_side)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    hsv = rgb_to_hsv_np(rgb)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    mask = (s >= sat_thresh) & (v >= val_thresh) & (alpha >= alpha_thresh)
    weights = (s * v) * mask.astype(np.float32)

    hue = circular_mean_deg(h, weights)
    is_gray = hue is None
    brightness = float(np.mean(v))
    return is_gray, hue, brightness


def safe_copy_atomic(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    while True:
        candidate = dst if k == 0 else dst.with_name(f"{dst.stem}__{k}{dst.suffix}")
        tmp = candidate.with_suffix(candidate.suffix + ".tmp")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, candidate)
            return candidate
        except FileExistsError:
            k += 1
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise


def main():
    parser = argparse.ArgumentParser(
        prog="colorsorter",
        description="Copy images from INPUT into a 'colorsorted' folder, named so that alphabetical order follows hue."
    )
    parser.add_argument("input", type=str, help="Input folder containing images")
    parser.add_argument("--out", type=str, default=None, help="Output folder (default: INPUT/colorsorted)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--max-side", type=int, default=512, help="Resize longest side to this for analysis (default 512)")
    parser.add_argument("--sat-thresh", type=float, default=0.15, help="Minimum saturation to consider a pixel colored (default 0.15)")
    parser.add_argument("--val-thresh", type=float, default=0.20, help="Minimum value (brightness) to consider (default 0.20)")
    parser.add_argument("--alpha-thresh", type=float, default=0.05, help="Minimum alpha to consider (default 0.05)")
    parser.add_argument("--gray-first", action="store_true", help="Place grayscale/low-sat images before colored ones")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy, just print planned order")
    parser.add_argument("--max-files", type=int, default=10000, help="Maximum number of images to process")
    parser.add_argument("--sandbox", type=str, default=None, help="Optional base directory; input/output must reside inside this path")
    parser.add_argument("--no-follow-symlinks", action="store_true", help="Do not follow symlinked files within the input tree")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else (in_dir / "colorsorted")

    if args.sandbox:
        sandbox = Path(args.sandbox).expanduser().resolve()
        if not is_in_dir(in_dir, sandbox):
            raise SystemExit(f"Input path must be inside sandbox: {sandbox}")
        if not is_in_dir(out_dir, sandbox):
            raise SystemExit(f"Output path must be inside sandbox: {sandbox}")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe = out_dir / ".colorsorter_write_test"
        with open(probe, "wb") as f:
            f.write(b"ok")
        probe.unlink(missing_ok=True)
    except Exception as e:
        raise SystemExit(f"Output directory not writable: {out_dir} ({e})")

    imgs = list_images(in_dir, recursive=args.recursive, exclude_dir=out_dir, follow_symlinks=not args.no_follow_symlinks)
    if not imgs:
        raise SystemExit("No images found. Supported: " + ", ".join(sorted(IMAGE_EXTS)))

    if len(imgs) > args.max_files:
        raise SystemExit(f"Too many images ({len(imgs)}). Use --max-files to raise the limit.")

    results = []
    for p in imgs:
        try:
            is_gray, hue, bright = image_hue_score(
                p, max_side=args.max_side, sat_thresh=args.sat_thresh, val_thresh=args.val_thresh, alpha_thresh=args.alpha_thresh
            )
            results.append((p, is_gray, hue, bright))
        except (PermissionError, FileNotFoundError) as e:
            logging.warning("Skipping %s: %s", p, e)
        except Exception as e:
            logging.warning("Skipping %s: %s", p, e)

    if not results:
        raise SystemExit("No valid images could be analyzed.")

    def sort_key(t):
        p, is_gray, hue, bright = t
        grp = (0 if is_gray else 1) if args.gray_first else (0 if not is_gray else 1)
        hkey = (int(round(hue)) if hue is not None else int(round(bright * 1000)))
        return (grp, hkey, p.name.lower())

    results.sort(key=sort_key)

    pad = max(3, len(str(len(results))))
    planned = []
    for idx, (p, is_gray, hue, bright) in enumerate(results, start=1):
        if is_gray or hue is None:
            htag = "hxxx"
        else:
            htag = f"h{int(round(hue))%360:03d}"
        new_name = f"{idx:0{pad}d}_{htag}_{p.name}"
        planned.append((p, out_dir / new_name))

    if args.dry_run:
        print(f"Planned copy to: {out_dir}")
        for src, dst in planned:
            print(f"{dst.name}  <=  {src.name}")
        return

    for src, dst in planned:
        try:
            final_path = safe_copy_atomic(src, dst)
            print(final_path.name)
        except Exception as e:
            logging.error("Failed to copy %s -> %s: %s", src, dst, e)

    print(f"Done. {len(planned)} file(s) written to: {out_dir}")


if __name__ == "__main__":
    main()
