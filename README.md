# colorsorter

Copy images into a `colorsorted/` folder so that alphabetical order follows each image’s dominant hue.

## Install

```bash
pip install pillow numpy
```

## Use

```bash
python colorsorter.py INPUT [--recursive] [--gray-first] [--dry-run] [-v]
# example
python colorsorter.py ~/Pictures --recursive -v
```

## Notes

* Writes to `INPUT/colorsorted` (or `--out`) without overwriting; uses atomic copies.
* Fast hue analysis via downscaled RGBA → HSV; thresholds tunable (`--sat-thresh`, `--val-thresh`, `--alpha-thresh`, `--max-side`).
* Optional grayscale-first sorting with flag '--gray-first'.
* Safety & control: `--sandbox PATH`, `--no-follow-symlinks`, `--max-files`, `--dry-run`.

* Supports: jpg/jpeg/png/bmp/tif/tiff/webp/gif.

