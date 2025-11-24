# Change Detection

An easy-to-use, advanced before/after image difference detector implemented in Python.

This repository contains `detection.py` — a lightweight pipeline to align, normalize, detect and visualize small/large changes between pairs of images. The script is optimized for real-world scenarios where lighting or camera alignment may vary.

---

## ✨ Key features

- Automatic alignment (feature matching + homography) to handle camera shifts/rotation
- Optional illumination compensation (histogram matching via scikit-image or CLAHE fallback)
- Robust difference calculation: SSIM (if available) or channel-aware absolute difference + adaptive thresholding
- Multi-scale (coarse + fine) detection for small crops and larger changes
- Morphological cleanup, connected-components filtering and merge-overlapping boxes
- Output: annotated image, difference mask, colored heatmap and JSON summary per pair

---

## 🔧 Requirements

- Python 3.8+ (recommended)
- OpenCV (cv2)
- NumPy
- Optional but recommended: scikit-image (enables SSIM and histogram matching)

Install with pip (Windows PowerShell examples):

```powershell
# minimal
python -m pip install --user opencv-python numpy

# recommended (adds scikit-image for higher-quality comparisons)
python -m pip install --user opencv-python numpy scikit-image
```

If you see a `ModuleNotFoundError: No module named 'skimage'`, installing `scikit-image` will enable better results (SSIM-based detection and accurate histogram matching). The script includes CLAHE-based fallbacks and will still run without `scikit-image`.

---

## 🧭 How the script expects your images

- Place input images in a single folder (default: `input-images`).
- Each *before* image must have a matching *after* image with the same base name suffixed with `~2` (e.g. `scene001.jpg` and `scene001~2.jpg`).
- Supported formats (by default): `.jpg`, `.png`.

Example layout:

```
input-images/
  scene001.jpg
  scene001~2.jpg
  scene002.jpg
  scene002~2.jpg
  ...
```

---

## ▶️ Quick start / Usage

Run the script with defaults (reads `input-images` and writes `output`):

```powershell
python -u "d:\assignments\change detection\detection.py"
```

Specify input and output folders explicitly:

```powershell
python -u "d:\assignments\change detection\detection.py" "path\to\input" "path\to\output"
```

Example with options:

```powershell
python -u detection.py input-images output --use_ssim --illumination_compensation --min_area 400 --morph_k 5
```

---

## 🛠️ Command-line options (summary)

- `input_folder`: (positional) folder containing before/after pairs (defaults to `input-images`)
- `output_folder`: (positional) folder to save results (defaults to `output`)
- `--threshold`: (legacy) integer threshold (default: 30)
- `--min_area`: minimum area in pixels for a region to be kept (default: 200)
- `--min_solidity`: minimum solidity for contours (default: 0.2)
- `--morph_k`: opening kernel size (integer, default: 3)
- `--close_k`: closing kernel size (integer, default: 7)
- `--use_ssim`: use SSIM when scikit-image is available (flag)
- `--illumination_compensation`: try histogram matching (scikit-image) or CLAHE fallback (flag)
- `--heat_alpha`: float 0..1 for heatmap overlay transparency (default: 0.55)
- `--max_features`: max features used for alignment (default: 3000)
- `--good_match_percent`: ratio of matches considered 'good' for homography (default: 0.15)
- `--iou_merge_thresh`: IoU threshold to merge nearby boxes (default: 0.2)

---

## 📦 Output structure and files

For each before/after pair `name.ext` (e.g. `scene001.jpg`) the script creates a subfolder in the output folder with the base name `name/` and writes:

- `annotated.ext` — the original "after" image with detected regions annotated
- `mask.png` — cleaned binary mask (0/255) of detected regions
- `heatmap.png` — colored heatmap overlay used to visualize per-pixel intensity
- `summary.json` — per-pair JSON report containing `before`, `after`, `aligned`, `num_changes`, `changes` (array of bounding box objects), and a timestamp

Additionally, a global file `all_summaries.json` is produced in the `output` folder summarizing all processed pairs.

Example `summary.json` snippet:

```json
{
  "before": "input-images/scene001.jpg",
  "after": "input-images/scene001~2.jpg",
  "aligned": true,
  "num_changes": 2,
  "changes": [ { "x": 10, "y": 20, "w": 100, "h": 50, "area": 500.0 }, ... ],
  "timestamp": "2025-11-25T14:12:33Z"
}
```

---

## 🔍 Internals — how detection works (high level)

1. Alignment: ORB feature matching + homography to align `after` onto `before`. If alignment fails, the script continues using the original `after` image.
2. Illumination compensation: If scikit-image is installed, the script attempts `match_histograms` (better) — otherwise it applies per-channel CLAHE.
3. Difference computation:
   - If scikit-image is available and `--use_ssim` is set: compute SSIM map (high-quality structural change map), then Otsu thresholding.
   - Otherwise: compute channel-wise absolute difference and a weighted combination (more weight to green), normalize and apply adaptive thresholding.
   - The script runs two passes (coarse: larger blur; fine: smaller blur) and unions the results for multi-scale detection.
4. Postprocess: morphological opening/closing, connected-component size filtering, contour solidity filtering.
5. Merge overlapping boxes (IoU based) and annotate/output results.

---

## 🎯 Tuning tips

- To reduce false positives (small specks): increase `--min_area` and/or increase `--morph_k`/`--close_k`.
- To improve detection of tiny changes: decrease `--min_area`, ensure `--use_ssim` is enabled, and try tuning `--good_match_percent` / `--max_features` for better alignment.
- If alignment fails often: increase `--max_features` or reduce `--good_match_percent` to allow more matches into homography estimation.
- Illumination issues: use `--illumination_compensation` to correct lighting between images.

---

## 🐞 Troubleshooting

- "ModuleNotFoundError: No module named 'skimage'"
  - Install scikit-image: `python -m pip install --user scikit-image` (recommended) or run without `--use_ssim`.
- Lots of small noise detections → bump `--min_area` to 500+ and/or increase morphological kernel sizes.
- Uploaded images not processed → ensure `after` images are named with `~2` suffix, same extension as `before`.

---

## ✅ Example runs

Run with best-effort (SSIM + illumination matching):

```powershell
python -u detection.py input-images output --use_ssim --illumination_compensation
```

Run minimal (no scikit-image required):

```powershell
python -u detection.py input-images output --min_area 300
```

---

## 📎 Notes & next steps

- The script is intentionally conservative with defaults so it works across many real-world datasets.
- Consider adding a small test dataset and a CLI `--debug` flag if you plan to extensively tune behaviour for a particular camera / scene.
- Add a `requirements.txt` or `pyproject.toml` if you want reproducible environments for CI / packaging.

---

## ⚖️ License & attribution

Author : **Tariful Islam**
