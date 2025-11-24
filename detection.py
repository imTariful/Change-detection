import os
import cv2
import numpy as np
import json
from glob import glob
import argparse
from datetime import datetime

_HAVE_SKIMAGE = False
try:
    from skimage.exposure import match_histograms
    from skimage.metrics import structural_similarity as ssim
    _HAVE_SKIMAGE = True
except Exception:
    # fallback: we'll use CLAHE + absdiff
    _HAVE_SKIMAGE = False

# ---------- Utilities ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

# ---------- Alignment ----------
def align_images(before, after, max_features=5000, good_match_percent=0.15):
    """
    Align 'after' to 'before' using feature matching + homography.
    Returns aligned_after, homography (or None).
    """
    # Convert to gray
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Use ORB (fast) and fallback to AKAZE if ORB fails
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(before_gray, None)
    kp2, des2 = orb.detectAndCompute(after_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        # fallback: return after as-is
        return after, None

    # Match descriptors.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return after, None

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(8, int(len(matches) * good_match_percent))
    matches = matches[:num_good]

    # Extract location of good matches
    pts_before = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_after = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(pts_after, pts_before, cv2.RANSAC, 5.0)
    if H is None:
        return after, None

    h, w = before.shape[:2]
    aligned_after = cv2.warpPerspective(after, H, (w, h), flags=cv2.INTER_LINEAR)
    return aligned_after, H

# ---------- Illumination compensation ----------
def match_illumination(before, after):
    """
    Try histogram matching (skimage) else apply CLAHE per channel.
    """
    if _HAVE_SKIMAGE:
        # skimage.match_histograms handles multichannel
        matched = match_histograms(after, before, multichannel=True)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return matched
    else:
        # CLAHE per channel
        lab_before = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
        lab_after = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_after = clahe.apply(lab_after[:, :, 0])
        lab_after[:, :, 0] = l_after
        matched = cv2.cvtColor(lab_after, cv2.COLOR_LAB2BGR)
        return matched

# ---------- Difference computation ----------
def compute_difference(before, after, use_ssim=True, gaussian_blur=3):
    """
    Returns diff_mask (uint8 0/255), diff_heatmap (float 0..1)
    """
    # Convert to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    if gaussian_blur and gaussian_blur > 0:
        before_gray = cv2.GaussianBlur(before_gray, (gaussian_blur, gaussian_blur), 0)
        after_gray = cv2.GaussianBlur(after_gray, (gaussian_blur, gaussian_blur), 0)

    if _HAVE_SKIMAGE and use_ssim:
        # structural similarity returns score and full diff image (range -1..1)
        score, diff = ssim(before_gray, after_gray, full=True)
        # convert to absolute diff heatmap 0..1
        diff = (1.0 - diff)  # higher => more different
        diff = np.clip(diff, 0, 1)
        diff_heatmap = diff.astype(np.float32)
        # threshold using Otsu on scaled diff
        diff_u8 = (diff_heatmap * 255).astype(np.uint8)
        _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask, diff_heatmap
    else:
        # Enhanced absolute difference with channel-aware combination
        diff_b = cv2.absdiff(before[:, :, 0], after[:, :, 0]).astype(np.float32)
        diff_g = cv2.absdiff(before[:, :, 1], after[:, :, 1]).astype(np.float32)
        diff_r = cv2.absdiff(before[:, :, 2], after[:, :, 2]).astype(np.float32)
        # weighted combination (human eye sensitive to green)
        diff_combined = 0.25 * diff_b + 0.5 * diff_g + 0.25 * diff_r
        diff_norm = (diff_combined / 255.0).astype(np.float32)
        # Adaptive threshold: local mean
        diff_u8 = (diff_norm * 255).astype(np.uint8)
        # Use adaptive threshold to capture both small/large changes
        mask_adapt = cv2.adaptiveThreshold(diff_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 21, -7)
        return mask_adapt, diff_norm

# ---------- Postprocessing ----------
def clean_mask(mask, min_area=100, morph_k=3, close_k=7):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = mask.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    # remove small components via connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def find_bounding_boxes(mask, min_area=100, min_solidity=0.2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else area
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue
        boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'area': float(area)})
    return boxes

def merge_overlapping_boxes(boxes, iou_thresh=0.2):
    """
    Merge boxes with IoU > iou_thresh.
    Boxes: list of dicts with x,y,w,h.
    """
    if not boxes:
        return []
    rects = np.array([[b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']] for b in boxes])
    areas = np.array([b['area'] for b in boxes])
    keep = []
    used = set()
    for i in range(len(rects)):
        if i in used:
            continue
        x1, y1, x2, y2 = rects[i].tolist()
        group = [i]
        used.add(i)
        for j in range(i + 1, len(rects)):
            if j in used:
                continue
            xx1 = max(x1, rects[j, 0])
            yy1 = max(y1, rects[j, 1])
            xx2 = min(x2, rects[j, 2])
            yy2 = min(y2, rects[j, 3])
            iw = max(0, xx2 - xx1)
            ih = max(0, yy2 - yy1)
            inter = iw * ih
            union = (x2 - x1) * (y2 - y1) + (rects[j, 2] - rects[j, 0]) * (rects[j, 3] - rects[j, 1]) - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_thresh:
                group.append(j)
                used.add(j)
        # merge group
        group_rects = rects[group]
        gx1 = int(np.min(group_rects[:, 0]))
        gy1 = int(np.min(group_rects[:, 1]))
        gx2 = int(np.max(group_rects[:, 2]))
        gy2 = int(np.max(group_rects[:, 3]))
        total_area = float(np.sum([areas[k] for k in group]))
        keep.append({'x': gx1, 'y': gy1, 'w': gx2 - gx1, 'h': gy2 - gy1, 'area': total_area})
    return keep

# ---------- Visualization ----------
def make_heatmap_overlay(base_img, diff_norm, alpha=0.6):
    """
    diff_norm: float array 0..1 same size as base_img
    """
    heat = (np.clip(diff_norm, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_img, 1.0 - alpha, heat_color, alpha, 0)
    return overlay, heat_color

def annotate_image(img, boxes, label_prefix="Change", color=(0, 0, 255)):
    out = img.copy()
    for i, b in enumerate(boxes, 1):
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        area = int(b.get('area', w * h))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{label_prefix} {i}: {area}"
        cv2.putText(out, text, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

# ---------- Main processing pipeline ----------
def process_pair(before_path, after_path, out_dir, args):
    filename = os.path.basename(before_path)
    name, ext = os.path.splitext(filename)
    out_base = os.path.join(out_dir, name)
    ensure_dir(out_base)

    before = read_image(before_path)
    after = read_image(after_path)

    # Align images
    aligned_after, H = align_images(before, after, max_features=args.max_features, good_match_percent=args.good_match_percent)
    aligned_flag = H is not None

    # Illumination compensation
    after_matched = match_illumination(before, aligned_after) if args.illumination_compensation else aligned_after

    # Coarse -> fine detection (you can tune gaussian blur sizes)
    mask_coarse, heat_coarse = compute_difference(before, after_matched, use_ssim=args.use_ssim, gaussian_blur=11)
    mask_fine, heat_fine = compute_difference(before, after_matched, use_ssim=args.use_ssim, gaussian_blur=3)

    # Combine masks: union and weight heatmaps
    combined_mask = cv2.bitwise_or(mask_coarse, mask_fine)
    # If masks returned as float/diff norm, ensure binary
    if combined_mask.dtype != np.uint8:
        combined_mask = (combined_mask * 255).astype(np.uint8)
    # Clean mask
    cleaned = clean_mask(combined_mask, min_area=args.min_area, morph_k=args.morph_k, close_k=args.close_k)

    # Find boxes and merge
    boxes = find_bounding_boxes(cleaned, min_area=args.min_area, min_solidity=args.min_solidity)
    boxes_merged = merge_overlapping_boxes(boxes, iou_thresh=args.iou_merge_thresh)

    # Create heatmap overlay (use average of coarse & fine heat)
    if isinstance(heat_coarse, np.ndarray) and isinstance(heat_fine, np.ndarray):
        heat = (0.5 * heat_coarse + 0.5 * heat_fine)
    else:
        # fallback: create heat from cleaned mask
        heat = (cleaned.astype(np.float32) / 255.0)

    overlay, heat_color = make_heatmap_overlay(after_matched, heat, alpha=args.heat_alpha)
    annotated = annotate_image(overlay, boxes_merged, label_prefix="Change", color=(0, 0, 255))

    # Save outputs
    annotated_path = os.path.join(out_base, f"annotated{ext}")
    mask_path = os.path.join(out_base, f"mask.png")
    heatmap_path = os.path.join(out_base, f"heatmap.png")
    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(mask_path, cleaned)
    cv2.imwrite(heatmap_path, heat_color)

    # Save JSON summary
    summary = {
        "before": before_path,
        "after": after_path,
        "aligned": bool(aligned_flag),
        "num_changes": len(boxes_merged),
        "changes": boxes_merged,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    summary_path = os.path.join(out_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# ---------- Batch processing ----------
def process_image_pairs(input_folder, output_folder, args):
    ensure_dir(output_folder)
    before_images = sorted(glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png")))
    # filter the "~2" after images from before_images list
    before_images = [p for p in before_images if "~2" not in os.path.basename(p)]
    summaries = []
    for before_path in before_images:
        filename = os.path.basename(before_path)
        name, ext = os.path.splitext(filename)
        after_filename = f"{name}~2{ext}"
        after_path = os.path.join(input_folder, after_filename)
        if not os.path.exists(after_path):
            print(f"[WARN] After image not found for {filename}, expected {after_filename}")
            continue
        try:
            summary = process_pair(before_path, after_path, output_folder, args)
            summaries.append(summary)
            print(f"[OK] Processed {filename} -> {len(summary['changes'])} changes")
        except Exception as e:
            print(f"[ERR] Failed {filename}: {e}")
    # Save global summary
    with open(os.path.join(output_folder, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    return summaries

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Advanced before/after image difference detector")
    p.add_argument("input_folder", nargs="?", default="input-images")
    p.add_argument("output_folder", nargs="?", default="output")
    p.add_argument("--threshold", type=int, default=30, help="(legacy) threshold")
    p.add_argument("--min_area", type=int, default=200, help="min area of detected region")
    p.add_argument("--min_solidity", type=float, default=0.2, help="minimum solidity to accept contour")
    p.add_argument("--morph_k", type=int, default=3, help="morph open kernel size")
    p.add_argument("--close_k", type=int, default=7, help="morph close kernel size")
    p.add_argument("--use_ssim", action="store_true", help="use SSIM if skimage available")
    p.add_argument("--illumination_compensation", action="store_true", help="attempt to match illumination")
    p.add_argument("--heat_alpha", type=float, default=0.55, help="heatmap overlay alpha")
    p.add_argument("--max_features", type=int, default=3000)
    p.add_argument("--good_match_percent", type=float, default=0.15)
    p.add_argument("--iou_merge_thresh", type=float, default=0.2)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    summaries = process_image_pairs(args.input_folder, args.output_folder, args)
    print(f"Done. Processed {len(summaries)} pairs.")