import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TOLERANCE = 20
DEFAULT_BIN_SIZE = 16
DEFAULT_TOP_K_COLORS = 8
EDGE_COVERAGE_THRESHOLD = 0.9
MIN_EDGE_RATIO = 0.01
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_FRAME_INTERVAL_SECONDS = 2.0
PROGRESS_BAR_WIDTH = 28
WARNING_MIN_FRAMES = 3
WARNING_STABLE_GROUP_MIN_FRAMES = 5
WARNING_AREA_RELATIVE_TOLERANCE = 0.03
WARNING_AREA_ABSOLUTE_TOLERANCE_PIXELS = 500
WARNING_DIMENSION_RELATIVE_TOLERANCE = 0.03
WARNING_DIMENSION_ABSOLUTE_TOLERANCE_PX = 6
MIN_L_PAIR_SCORE = 0.35
MIN_RECTANGULARITY_FOR_RECT = 0.9
FULL_FRAME_AD_THRESHOLD_PERCENT = 99.5
MIN_AD_AREA_PERCENT = 5.0
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class NoAdsDetectedError(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a single-color ad region and report its size and area percentage."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["images", "videos"],
        help="Choose the batch input source under input/: images or videos.",
    )
    parser.add_argument(
        "--color",
        help="Optional ad color in hex format, for example #FFA223 or FFA223.",
    )
    parser.add_argument(
        "--shape",
        choices=["auto", "l", "square", "rectangle"],
        default="auto",
        help="Optional shape hint. Use auto to infer from the mask.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help="Max RGB distance used to match the solid ad color.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_SECONDS,
        help=f"Frame capture interval in seconds for video mode. Default: {DEFAULT_FRAME_INTERVAL_SECONDS}",
    )
    return parser.parse_args()


def parse_hex_color(color_text):
    value = color_text.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Color must be in RRGGBB format.")
    try:
        rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError("Color must be a valid hex value.") from exc
    return rgb


def rgb_to_hex(color_rgb):
    return "#{:02X}{:02X}{:02X}".format(*color_rgb)


def load_image_rgb(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_border_pixels(image_rgb):
    height, width = image_rgb.shape[:2]
    return np.concatenate(
        [
            image_rgb[0, :, :],
            image_rgb[height - 1, :, :],
            image_rgb[:, 0, :],
            image_rgb[:, width - 1, :],
        ],
        axis=0,
    )


def get_all_pixels(image_rgb):
    return image_rgb.reshape(-1, 3)


def quantized_top_colors(pixels, top_k=DEFAULT_TOP_K_COLORS, bin_size=DEFAULT_BIN_SIZE):
    quantized = (pixels // bin_size) * bin_size
    counts = Counter(map(tuple, quantized.tolist()))
    candidates = []

    for color_key, _ in counts.most_common(top_k):
        color_key = np.array(color_key)
        mask = np.all(quantized == color_key, axis=1)
        median_color = np.median(pixels[mask], axis=0)
        candidate = tuple(int(round(value)) for value in median_color.tolist())
        candidates.append(candidate)

    return candidates


def deduplicate_colors(colors):
    seen = set()
    unique_colors = []
    for color in colors:
        if color in seen:
            continue
        seen.add(color)
        unique_colors.append(color)
    return unique_colors


def auto_color_candidates(image_rgb, top_k=DEFAULT_TOP_K_COLORS, bin_size=DEFAULT_BIN_SIZE):
    border_candidates = quantized_top_colors(
        get_border_pixels(image_rgb),
        top_k=top_k,
        bin_size=bin_size,
    )
    full_image_candidates = quantized_top_colors(
        get_all_pixels(image_rgb),
        top_k=top_k,
        bin_size=bin_size,
    )
    return deduplicate_colors(border_candidates + full_image_candidates)


def build_color_mask(image_rgb, color_rgb, tolerance):
    image_int = image_rgb.astype(np.int32)
    color_int = np.array(color_rgb, dtype=np.int32)
    diff = image_int - color_int
    dist_squared = np.sum(diff * diff, axis=2)
    mask = (dist_squared <= tolerance * tolerance).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def is_border_component(component_mask):
    return (
        component_mask[0, :].any()
        or component_mask[-1, :].any()
        or component_mask[:, 0].any()
        or component_mask[:, -1].any()
    )


def count_consecutive(values, from_start, threshold=EDGE_COVERAGE_THRESHOLD):
    count = 0
    sequence = values if from_start else values[::-1]
    for value in sequence:
        if value >= threshold:
            count += 1
        else:
            break
    return count


def infer_shape_from_component(component_crop, bbox_width, bbox_height, area_pixels):
    column_coverage = component_crop.mean(axis=0)
    row_coverage = component_crop.mean(axis=1)

    edges = {
        "left": count_consecutive(column_coverage, from_start=True),
        "right": count_consecutive(column_coverage, from_start=False),
        "top": count_consecutive(row_coverage, from_start=True),
        "bottom": count_consecutive(row_coverage, from_start=False),
    }

    rectangularity = area_pixels / float(bbox_width * bbox_height)
    aspect_gap = abs(bbox_width - bbox_height) / float(max(bbox_width, bbox_height))

    adjacent_pairs = [
        ("left", "top"),
        ("top", "right"),
        ("right", "bottom"),
        ("bottom", "left"),
    ]

    min_vertical = max(1, int(round(bbox_width * MIN_EDGE_RATIO)))
    min_horizontal = max(1, int(round(bbox_height * MIN_EDGE_RATIO)))
    pair_scores = []

    for edge_a, edge_b in adjacent_pairs:
        edge_a_min = min_vertical if edge_a in ("left", "right") else min_horizontal
        edge_b_min = min_vertical if edge_b in ("left", "right") else min_horizontal
        if edges[edge_a] < edge_a_min or edges[edge_b] < edge_b_min:
            continue

        edge_a_norm = edges[edge_a] / float(bbox_width if edge_a in ("left", "right") else bbox_height)
        edge_b_norm = edges[edge_b] / float(bbox_width if edge_b in ("left", "right") else bbox_height)
        pair_scores.append((edge_a_norm + edge_b_norm, edge_a, edge_b))

    best_pair = None
    if pair_scores:
        pair_scores.sort(reverse=True)
        best_pair_score, edge_a, edge_b = pair_scores[0]
        best_pair = (edge_a, edge_b)
    else:
        best_pair_score = 0.0

    if rectangularity < MIN_RECTANGULARITY_FOR_RECT and best_pair and best_pair_score >= MIN_L_PAIR_SCORE:
        return "l", edges, best_pair, rectangularity, best_pair_score

    if rectangularity >= MIN_RECTANGULARITY_FOR_RECT and aspect_gap <= 0.05:
        return "square", edges, best_pair, rectangularity, best_pair_score

    if rectangularity >= MIN_RECTANGULARITY_FOR_RECT:
        return "rectangle", edges, best_pair, rectangularity, best_pair_score

    return "unknown", edges, best_pair, rectangularity, best_pair_score


def compute_component_score(
    area_pixels,
    inferred_shape,
    rectangularity,
    best_pair_score,
    bbox_width,
    bbox_height,
    shape_hint,
):
    aspect_gap = abs(bbox_width - bbox_height) / float(max(bbox_width, bbox_height))

    if shape_hint == "l":
        shape_score = best_pair_score
        if inferred_shape == "l":
            shape_score += 1.0
        else:
            shape_score *= 0.25
    elif shape_hint == "square":
        square_like = max(0.0, 1.0 - min(aspect_gap, 0.25) / 0.25)
        shape_score = rectangularity + square_like
        if inferred_shape == "square":
            shape_score += 1.0
        elif inferred_shape == "rectangle":
            shape_score += 0.25
        else:
            shape_score *= 0.25
    elif shape_hint == "rectangle":
        rectangle_like = rectangularity + max(0.0, min(aspect_gap, 0.2))
        if inferred_shape == "rectangle":
            shape_score = rectangle_like + 1.0
        elif inferred_shape == "square":
            shape_score = rectangle_like + 0.75
        else:
            shape_score = rectangle_like * 0.25
    elif inferred_shape == "unknown":
        shape_score = 0.0
    else:
        shape_score = max(rectangularity, best_pair_score)
        if inferred_shape == "l":
            shape_score += 0.15

    return area_pixels * max(shape_score, 1e-6)


def select_best_component(image_rgb, candidate_colors, tolerance, shape_hint):
    best = None

    for color_rgb in candidate_colors:
        mask = build_color_mask(image_rgb, color_rgb, tolerance)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 500:
                continue

            x, y, width, height, _ = stats[label]
            component_mask = labels == label
            component_crop = component_mask[y : y + height, x : x + width]
            inferred_shape, edges, edge_pair, rectangularity, best_pair_score = (
                infer_shape_from_component(
                    component_crop=component_crop,
                    bbox_width=width,
                    bbox_height=height,
                    area_pixels=area,
                )
            )
            if inferred_shape == "unknown":
                continue
            score = compute_component_score(
                area_pixels=area,
                inferred_shape=inferred_shape,
                rectangularity=rectangularity,
                best_pair_score=best_pair_score,
                bbox_width=width,
                bbox_height=height,
                shape_hint=shape_hint,
            )

            if best is None or score > best["selection_score"]:
                best = {
                    "color_rgb": color_rgb,
                    "labels": labels,
                    "label": label,
                    "stats": stats,
                    "area_pixels": area,
                    "selection_score": score,
                    "inferred_shape": inferred_shape,
                    "edges": edges,
                    "edge_pair": edge_pair,
                    "rectangularity": rectangularity,
                    "best_pair_score": best_pair_score,
                    "touches_border": is_border_component(component_mask),
                }

    return best


def normalize_l_pair(edge_pair):
    edge_set = set(edge_pair)
    if edge_set == {"left", "top"}:
        return ("left", "top")
    if edge_set == {"top", "right"}:
        return ("right", "top")
    if edge_set == {"right", "bottom"}:
        return ("right", "bottom")
    return ("left", "bottom")


def l_corner_name(edge_pair):
    edge_a, edge_b = normalize_l_pair(edge_pair)
    corner_lookup = {
        ("left", "top"): "top-left",
        ("right", "top"): "top-right",
        ("right", "bottom"): "bottom-right",
        ("left", "bottom"): "bottom-left",
    }
    return corner_lookup[(edge_a, edge_b)]


def resolve_shape(shape_hint, inferred_shape):
    if shape_hint == "auto":
        return inferred_shape
    return shape_hint


def build_dimensions(shape_name, bbox, edges, edge_pair):
    bbox_width = bbox["width"]
    bbox_height = bbox["height"]

    if shape_name == "l":
        if edge_pair is None:
            vertical_edge = "left" if edges["left"] >= edges["right"] else "right"
            horizontal_edge = "top" if edges["top"] >= edges["bottom"] else "bottom"
            edge_pair = (vertical_edge, horizontal_edge)

        edge_a, edge_b = normalize_l_pair(edge_pair)
        dimensions = {
            "corner": l_corner_name((edge_a, edge_b)),
        }

        if edge_a in ("left", "right"):
            dimensions[f"{edge_a}_arm"] = {
                "width_px": edges[edge_a],
                "height_px": bbox_height,
            }
        else:
            dimensions[f"{edge_a}_arm"] = {
                "width_px": bbox_width,
                "height_px": edges[edge_a],
            }

        if edge_b in ("left", "right"):
            dimensions[f"{edge_b}_arm"] = {
                "width_px": edges[edge_b],
                "height_px": bbox_height,
            }
        else:
            dimensions[f"{edge_b}_arm"] = {
                "width_px": bbox_width,
                "height_px": edges[edge_b],
            }

        return dimensions

    return {
        "width_px": bbox_width,
        "height_px": bbox_height,
    }


def input_source_dir(source_name):
    return Path(DEFAULT_INPUT_DIR) / source_name


def output_source_dir(source_name):
    return Path(DEFAULT_OUTPUT_DIR) / source_name


def detection_output_path(image_path, output_dir):
    return output_dir / f"{image_path.stem}_detected.png"


def warning_output_path(output_path):
    stem = output_path.stem.removesuffix("_detected")
    return output_path.with_name(f"{stem}_warning{output_path.suffix}")


def no_ad_output_path(image_path, output_dir):
    return output_dir / f"{image_path.stem}_no_ad.png"


def output_variant_paths(image_path, output_dir):
    detected_path = detection_output_path(image_path, output_dir)
    return [
        detected_path,
        warning_output_path(detected_path),
        no_ad_output_path(image_path, output_dir),
        detected_path.with_name(f"{detected_path.stem}_warning{detected_path.suffix}"),
    ]


def cleanup_output_variants(image_path, output_dir, keep_path=None):
    keep_resolved = keep_path.resolve() if keep_path is not None else None
    for candidate_path in output_variant_paths(image_path, output_dir):
        if keep_resolved is not None and candidate_path.resolve() == keep_resolved:
            continue
        if candidate_path.exists():
            candidate_path.unlink()


def is_supported_image_file(path):
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not path.stem.endswith("_detected")
    )


def is_supported_video_file(path):
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def find_source_files(source_dir, source_name):
    if source_name == "images":
        return sorted(path for path in source_dir.iterdir() if is_supported_image_file(path))
    return sorted(path for path in source_dir.iterdir() if is_supported_video_file(path))


def format_timestamp_tag(seconds):
    return f"{seconds:08.2f}".replace(".", "_")


def print_progress(label, current, total):
    total = max(0, int(total))
    current = max(0, min(int(current), total if total > 0 else int(current)))

    if total <= 0:
        print(f"\r{label}: {current}", end="", flush=True)
        return

    ratio = current / float(total)
    filled = int(round(PROGRESS_BAR_WIDTH * ratio))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    percent = int(round(ratio * 100))
    print(
        f"\r{label}: [{bar}] {current}/{total} ({percent}%)",
        end="",
        flush=True,
    )


def finish_progress(total):
    if total > 0:
        print()


def extract_frames_from_video(video_path, frames_dir, frame_interval_seconds):
    if frame_interval_seconds <= 0:
        raise ValueError("--frame-interval must be greater than 0.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 1.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_step = max(1, int(round(fps * frame_interval_seconds)))
    extracted_frames = []
    frame_index = 0
    next_capture_index = 0
    capture_index = 0

    while True:
        success, frame_bgr = capture.read()
        if not success:
            break

        if capture_index >= next_capture_index:
            timestamp_seconds = capture_index / fps
            frame_name = (
                f"{video_path.stem}_frame_{frame_index:04d}_t{format_timestamp_tag(timestamp_seconds)}s.png"
            )
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame_bgr)
            extracted_frames.append(
                {
                    "frame_path": frame_path,
                    "frame_index": capture_index,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                }
            )
            frame_index += 1
            next_capture_index += frame_step

        capture_index += 1
        print_progress(
            label=f"Extracting {video_path.stem}",
            current=capture_index,
            total=total_frames,
        )

    capture.release()
    finish_progress(total_frames)

    if not extracted_frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return extracted_frames


def numeric_median(values):
    if not values:
        return 0
    return float(np.median(np.array(values, dtype=np.float64)))


def dimension_metric_values(result):
    metrics = {}
    dimensions = result["dimensions"]
    if result["shape"] == "l":
        metrics["corner"] = dimensions["corner"]
        for arm_name, arm_size in dimensions.items():
            if arm_name == "corner":
                continue
            metrics[f"{arm_name}_width_px"] = arm_size["width_px"]
            metrics[f"{arm_name}_height_px"] = arm_size["height_px"]
        return metrics

    metrics["width_px"] = dimensions["width_px"]
    metrics["height_px"] = dimensions["height_px"]
    return metrics


def dominant_value(values):
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def build_video_warning_reference(results):
    if len(results) < WARNING_MIN_FRAMES:
        return None

    dominant_shape = dominant_value([result["shape"] for result in results])
    dominant_shape_results = [result for result in results if result["shape"] == dominant_shape]
    if len(dominant_shape_results) < 2:
        return None

    reference = {
        "shape": dominant_shape,
        "area_pixels": numeric_median([result["area_pixels"] for result in dominant_shape_results]),
        "metrics": {},
    }

    metric_maps = [dimension_metric_values(result) for result in dominant_shape_results]
    metric_keys = set(metric_maps[0].keys())
    for metric_map in metric_maps[1:]:
        metric_keys &= set(metric_map.keys())

    for metric_key in sorted(metric_keys):
        values = [metric_map[metric_key] for metric_map in metric_maps]
        if isinstance(values[0], str):
            reference["metrics"][metric_key] = dominant_value(values)
        else:
            reference["metrics"][metric_key] = numeric_median(values)

    return reference


def build_warning_reference_from_results(results):
    if not results:
        return None

    reference = {
        "shape": dominant_value([result["shape"] for result in results]),
        "area_pixels": numeric_median([result["area_pixels"] for result in results]),
        "area_percent": numeric_median([result["area_percent"] for result in results]),
        "frame_count": len(results),
        "metrics": {},
    }

    metric_maps = [dimension_metric_values(result) for result in results]
    metric_keys = set(metric_maps[0].keys())
    for metric_map in metric_maps[1:]:
        metric_keys &= set(metric_map.keys())

    for metric_key in sorted(metric_keys):
        values = [metric_map[metric_key] for metric_map in metric_maps]
        if isinstance(values[0], str):
            reference["metrics"][metric_key] = dominant_value(values)
        else:
            reference["metrics"][metric_key] = numeric_median(values)

    return reference


def area_matches_reference(area_pixels, baseline_pixels):
    return not numeric_difference_exceeds_tolerance(
        value=area_pixels,
        baseline=baseline_pixels,
        absolute_tolerance=WARNING_AREA_ABSOLUTE_TOLERANCE_PIXELS,
        relative_tolerance=WARNING_AREA_RELATIVE_TOLERANCE,
    )


def cluster_results_by_area(results):
    clusters = []

    for result in sorted(results, key=lambda item: (item["shape"], item["area_pixels"])):
        matched_cluster = None
        for cluster in clusters:
            if cluster["shape"] != result["shape"]:
                continue
            if area_matches_reference(result["area_pixels"], cluster["area_pixels"]):
                matched_cluster = cluster
                break

        if matched_cluster is None:
            matched_cluster = {
                "shape": result["shape"],
                "area_pixels": result["area_pixels"],
                "results": [],
            }
            clusters.append(matched_cluster)

        matched_cluster["results"].append(result)
        matched_cluster["area_pixels"] = numeric_median(
            [item["area_pixels"] for item in matched_cluster["results"]]
        )

    return clusters


def build_stable_warning_references(results):
    if len(results) < WARNING_MIN_FRAMES:
        return []

    stable_references = []
    for cluster in cluster_results_by_area(results):
        if len(cluster["results"]) < WARNING_STABLE_GROUP_MIN_FRAMES:
            continue
        reference = build_warning_reference_from_results(cluster["results"])
        if reference is not None:
            stable_references.append(reference)

    return stable_references


def match_warning_reference(result, references):
    matched_reference = None
    best_distance = None

    for reference in references:
        if result["shape"] != reference["shape"]:
            continue
        if not area_matches_reference(result["area_pixels"], reference["area_pixels"]):
            continue

        distance = abs(result["area_pixels"] - reference["area_pixels"])
        if matched_reference is None or distance < best_distance:
            matched_reference = reference
            best_distance = distance

    return matched_reference


def numeric_difference_exceeds_tolerance(value, baseline, absolute_tolerance, relative_tolerance):
    tolerance = max(absolute_tolerance, abs(baseline) * relative_tolerance)
    return abs(value - baseline) > tolerance


def warning_reasons_for_result(result, reference):
    if reference is None:
        return []

    reasons = []
    if result["shape"] != reference["shape"]:
        reasons.append(
            f"shape khac voi da so frame ({result['shape']} vs {reference['shape']})"
        )
        return reasons

    if numeric_difference_exceeds_tolerance(
        value=result["area_pixels"],
        baseline=reference["area_pixels"],
        absolute_tolerance=WARNING_AREA_ABSOLUTE_TOLERANCE_PIXELS,
        relative_tolerance=WARNING_AREA_RELATIVE_TOLERANCE,
    ):
        reasons.append(
            f"dien tich khac biet ({result['area_pixels']} px vs median {int(round(reference['area_pixels']))} px)"
        )

    metric_values = dimension_metric_values(result)
    for metric_name, baseline_value in reference["metrics"].items():
        current_value = metric_values.get(metric_name)
        if current_value is None:
            continue

        if isinstance(baseline_value, str):
            if current_value != baseline_value:
                reasons.append(
                    f"{metric_name} khac biet ({current_value} vs {baseline_value})"
                )
            continue

        if numeric_difference_exceeds_tolerance(
            value=current_value,
            baseline=baseline_value,
            absolute_tolerance=WARNING_DIMENSION_ABSOLUTE_TOLERANCE_PX,
            relative_tolerance=WARNING_DIMENSION_RELATIVE_TOLERANCE,
        ):
            reasons.append(
                f"{metric_name} khac biet ({current_value} px vs median {int(round(baseline_value))} px)"
            )

    return reasons


def warning_reasons_without_matching_group(result, stable_references):
    if not stable_references:
        return []

    candidate_shapes = sorted({reference["shape"] for reference in stable_references})
    shape_text = ", ".join(candidate_shapes)
    return [
        "khong thuoc nhom on dinh co it nhat "
        f"{WARNING_STABLE_GROUP_MIN_FRAMES} frame cung ty le dien tich "
        f"(cac nhom hien co: {shape_text})"
    ]


def apply_video_frame_warnings(video_summary):
    stable_references = build_stable_warning_references(video_summary["results"])
    reference = None if stable_references else build_video_warning_reference(video_summary["results"])
    video_summary["warning_count"] = 0
    video_summary["warning_reference"] = reference
    video_summary["warning_reference_groups"] = stable_references

    for result in video_summary["results"]:
        matched_reference = match_warning_reference(result, stable_references)
        if matched_reference is not None:
            reasons = warning_reasons_for_result(result, matched_reference)
        elif stable_references:
            reasons = warning_reasons_without_matching_group(result, stable_references)
        else:
            reasons = warning_reasons_for_result(result, reference)
        result["warning_reasons"] = reasons
        result["is_warning"] = len(reasons) > 0

        current_output_path = Path(result["output_image_path"])
        flagged_output_path = warning_output_path(current_output_path)

        if result["is_warning"]:
            if current_output_path.exists():
                if flagged_output_path.exists():
                    flagged_output_path.unlink()
                current_output_path.replace(flagged_output_path)
            result["output_image_path"] = str(flagged_output_path)
            video_summary["warning_count"] += 1
        else:
            if flagged_output_path.exists():
                flagged_output_path.unlink()

    return video_summary


def choose_annotation_color(ad_color_rgb):
    candidate_colors_rgb = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
        (255, 255, 255),
    ]
    ad_color = np.array(ad_color_rgb, dtype=np.int32)
    chosen_rgb = max(
        candidate_colors_rgb,
        key=lambda color: np.sum((np.array(color, dtype=np.int32) - ad_color) ** 2),
    )
    return (chosen_rgb[2], chosen_rgb[1], chosen_rgb[0])


def fit_text_scale(text, max_width, max_height, thickness=2, min_scale=0.35, max_scale=1.2):
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    (base_width, base_height), base_baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        thickness,
    )
    base_height += base_baseline

    width_scale = max_width / float(max(base_width, 1))
    height_scale = max_height / float(max(base_height, 1))
    return max(min_scale, min(max_scale, width_scale, height_scale))


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def draw_text_box(
    image_bgr,
    text,
    center,
    max_width,
    max_height,
    border_color_bgr,
    background_color_bgr=(20, 20, 20),
    text_color_bgr=(255, 255, 255),
):
    scale = fit_text_scale(text, max_width=max_width, max_height=max_height)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    padding = max(4, int(round(scale * 8)))

    text_x = int(round(center[0] - text_width / 2.0))
    text_y = int(round(center[1] + text_height / 2.0))

    box_left = clamp(text_x - padding, 0, max(0, image_bgr.shape[1] - (text_width + padding * 2)))
    box_top = clamp(text_y - text_height - padding, 0, max(0, image_bgr.shape[0] - (text_height + baseline + padding * 2)))
    box_right = box_left + text_width + padding * 2
    box_bottom = box_top + text_height + baseline + padding * 2

    text_x = box_left + padding
    text_y = box_top + padding + text_height

    cv2.rectangle(
        image_bgr,
        (box_left, box_top),
        (box_right, box_bottom),
        background_color_bgr,
        thickness=-1,
    )
    cv2.rectangle(
        image_bgr,
        (box_left, box_top),
        (box_right, box_bottom),
        border_color_bgr,
        thickness=2,
    )
    cv2.putText(
        image_bgr,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def draw_measurement_line(image_bgr, start_point, end_point, color_bgr):
    cv2.line(image_bgr, start_point, end_point, color_bgr, thickness=2, lineType=cv2.LINE_AA)

    tick_size = 7
    if start_point[1] == end_point[1]:
        for point in (start_point, end_point):
            cv2.line(
                image_bgr,
                (point[0], point[1] - tick_size),
                (point[0], point[1] + tick_size),
                color_bgr,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
    else:
        for point in (start_point, end_point):
            cv2.line(
                image_bgr,
                (point[0] - tick_size, point[1]),
                (point[0] + tick_size, point[1]),
                color_bgr,
                thickness=2,
                lineType=cv2.LINE_AA,
            )


def apply_mask_overlay(image_bgr, component_mask, color_bgr, alpha=0.28):
    color_array = np.array(color_bgr, dtype=np.float32)
    mask = component_mask.astype(bool)
    image_bgr[mask] = (
        image_bgr[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha
    ).astype(np.uint8)


def draw_component_contours(image_bgr, component_mask, color_bgr):
    contour_mask = (component_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_bgr, contours, -1, color_bgr, thickness=3, lineType=cv2.LINE_AA)


def best_label_center(component_mask):
    mask_uint8 = (component_mask.astype(np.uint8)) * 255
    distance_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, _, _, max_location = cv2.minMaxLoc(distance_map)
    return max_location


def draw_area_label(image_bgr, component_mask, bbox, area_percent, border_color_bgr, shape_name):
    if shape_name in ("rectangle", "square"):
        center_x = bbox["x"] + bbox["width"] // 2
        center_y = bbox["y"] + bbox["height"] // 2
    else:
        center_x, center_y = best_label_center(component_mask)

    text = f"{area_percent:.3f}%"
    max_width = max(100, int(round(bbox["width"] * 0.45)))
    max_height = max(36, int(round(bbox["height"] * 0.22)))
    draw_text_box(
        image_bgr=image_bgr,
        text=text,
        center=(center_x, center_y),
        max_width=max_width,
        max_height=max_height,
        border_color_bgr=border_color_bgr,
    )


def save_no_ad_image(image_path, output_path):
    image_rgb = load_image_rgb(image_path)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_height, image_width = image_bgr.shape[:2]

    draw_text_box(
        image_bgr=image_bgr,
        text="No ads dection",
        center=(image_width // 2, max(32, image_height // 2)),
        max_width=max(180, int(round(image_width * 0.7))),
        max_height=max(40, int(round(image_height * 0.18))),
        border_color_bgr=(0, 0, 255),
        background_color_bgr=(32, 32, 32),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def draw_rectangle_dimensions(image_bgr, bbox, border_color_bgr):
    x = bbox["x"]
    y = bbox["y"]
    width = bbox["width"]
    height = bbox["height"]
    image_height, image_width = image_bgr.shape[:2]
    offset = max(18, int(round(min(image_width, image_height) * 0.03)))

    width_line_y = y - offset if y - offset >= 20 else min(image_height - 20, y + offset)
    height_line_x = x - offset if x - offset >= 20 else min(image_width - 20, x + offset)

    draw_measurement_line(
        image_bgr,
        start_point=(x, width_line_y),
        end_point=(x + width, width_line_y),
        color_bgr=border_color_bgr,
    )
    draw_text_box(
        image_bgr=image_bgr,
        text=f"W: {width} px",
        center=(x + width // 2, width_line_y - 18 if width_line_y <= y else width_line_y + 18),
        max_width=max(100, int(round(width * 0.8))),
        max_height=40,
        border_color_bgr=border_color_bgr,
    )

    draw_measurement_line(
        image_bgr,
        start_point=(height_line_x, y),
        end_point=(height_line_x, y + height),
        color_bgr=border_color_bgr,
    )
    draw_text_box(
        image_bgr=image_bgr,
        text=f"H: {height} px",
        center=(
            height_line_x - 85 if height_line_x <= x else height_line_x + 85,
            y + height // 2,
        ),
        max_width=180,
        max_height=max(40, int(round(height * 0.4))),
        border_color_bgr=border_color_bgr,
    )


def draw_l_dimensions(image_bgr, bbox, edge_pair, dimensions, border_color_bgr):
    if edge_pair is None:
        vertical_edge = "left" if "left_arm" in dimensions else "right"
        horizontal_edge = "top" if "top_arm" in dimensions else "bottom"
    else:
        vertical_edge, horizontal_edge = normalize_l_pair(edge_pair)
    image_height, image_width = image_bgr.shape[:2]

    vertical_width = dimensions[f"{vertical_edge}_arm"]["width_px"]
    vertical_height = dimensions[f"{vertical_edge}_arm"]["height_px"]
    horizontal_width = dimensions[f"{horizontal_edge}_arm"]["width_px"]
    horizontal_height = dimensions[f"{horizontal_edge}_arm"]["height_px"]

    y_line = bbox["y"] + bbox["height"] // 2
    if vertical_edge == "left":
        x_start = bbox["x"]
        x_end = bbox["x"] + vertical_width
    else:
        x_start = bbox["x"] + bbox["width"] - vertical_width
        x_end = bbox["x"] + bbox["width"]

    draw_measurement_line(
        image_bgr,
        start_point=(x_start, y_line),
        end_point=(x_end, y_line),
        color_bgr=border_color_bgr,
    )
    draw_text_box(
        image_bgr=image_bgr,
        text=f"{vertical_edge.title()} arm: {vertical_width} x {vertical_height} px",
        center=(
            (x_start + x_end) // 2,
            clamp(y_line - 24, 24, image_height - 24),
        ),
        max_width=max(120, min(int(round(image_width * 0.55)), bbox["width"])),
        max_height=42,
        border_color_bgr=border_color_bgr,
    )

    x_line = bbox["x"] + bbox["width"] // 2
    if horizontal_edge == "top":
        y_start = bbox["y"]
        y_end = bbox["y"] + horizontal_height
    else:
        y_start = bbox["y"] + bbox["height"] - horizontal_height
        y_end = bbox["y"] + bbox["height"]

    draw_measurement_line(
        image_bgr,
        start_point=(x_line, y_start),
        end_point=(x_line, y_end),
        color_bgr=border_color_bgr,
    )
    draw_text_box(
        image_bgr=image_bgr,
        text=f"{horizontal_edge.title()} arm: {horizontal_width} x {horizontal_height} px",
        center=(
            clamp(x_line + 150, 100, image_width - 100),
            (y_start + y_end) // 2,
        ),
        max_width=max(120, min(int(round(image_width * 0.6)), bbox["width"])),
        max_height=42,
        border_color_bgr=border_color_bgr,
    )


def save_annotated_image(image_rgb, component_mask, result, edge_pair, output_path):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    border_color_bgr = choose_annotation_color(parse_hex_color(result["detected_color_hex"]))

    apply_mask_overlay(image_bgr, component_mask, color_bgr=border_color_bgr)
    draw_component_contours(image_bgr, component_mask, color_bgr=border_color_bgr)

    if result["shape"] == "l":
        draw_l_dimensions(
            image_bgr=image_bgr,
            bbox=result["bbox"],
            edge_pair=edge_pair,
            dimensions=result["dimensions"],
            border_color_bgr=border_color_bgr,
        )
    else:
        draw_rectangle_dimensions(
            image_bgr=image_bgr,
            bbox=result["bbox"],
            border_color_bgr=border_color_bgr,
        )

    draw_area_label(
        image_bgr=image_bgr,
        component_mask=component_mask,
        bbox=result["bbox"],
        area_percent=result["area_percent"],
        border_color_bgr=border_color_bgr,
        shape_name=result["shape"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def detect_ad(image_path, color_hint, shape_hint, tolerance):
    image_rgb = load_image_rgb(image_path)
    image_width = image_rgb.shape[1]
    image_height = image_rgb.shape[0]

    if color_hint:
        candidate_colors = [parse_hex_color(color_hint)]
    else:
        candidate_colors = auto_color_candidates(image_rgb)

    best_component = select_best_component(
        image_rgb=image_rgb,
        candidate_colors=candidate_colors,
        tolerance=tolerance,
        shape_hint=shape_hint,
    )
    if best_component is None:
        raise NoAdsDetectedError(
            "No ads dection: detected region is not a supported shape."
        )

    label = best_component["label"]
    stats = best_component["stats"]
    x, y, width, height, area_pixels = stats[label]
    component_mask = best_component["labels"] == label
    inferred_shape = best_component["inferred_shape"]
    edges = best_component["edges"]
    edge_pair = best_component["edge_pair"]
    rectangularity = best_component["rectangularity"]
    resolved_shape = resolve_shape(shape_hint, inferred_shape)

    bbox = {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }
    area_percent = float(area_pixels) / float(image_width * image_height) * 100.0
    if area_percent < MIN_AD_AREA_PERCENT:
        raise NoAdsDetectedError(
            f"No ads dection: detected ad area is below {MIN_AD_AREA_PERCENT:.0f}% of the frame."
        )
    if area_percent >= FULL_FRAME_AD_THRESHOLD_PERCENT:
        raise NoAdsDetectedError(
            "No ads dection: detected ad covers the whole frame."
        )
    result = {
        "image_path": str(image_path),
        "detected_color_hex": rgb_to_hex(best_component["color_rgb"]),
        "shape": resolved_shape,
        "inferred_shape": inferred_shape,
        "bbox": bbox,
        "area_pixels": int(area_pixels),
        "area_percent": round(area_percent, 3),
        "dimensions": build_dimensions(resolved_shape, bbox, edges, edge_pair),
        "tolerance": tolerance,
        "rectangularity": round(rectangularity, 4),
        "selection_score": round(best_component["selection_score"], 3),
    }

    if shape_hint != "auto" and shape_hint != inferred_shape:
        result["warning"] = (
            f"Shape hint '{shape_hint}' differs from inferred shape '{inferred_shape}'."
        )

    render_data = {
        "image_rgb": image_rgb,
        "component_mask": component_mask,
        "edge_pair": edge_pair,
    }
    return result, render_data


def print_human_readable(result):
    print(f"Image: {result['image_path']}")
    print(f"Detected color: {result['detected_color_hex']}")
    print(f"Shape: {result['shape']}")
    print(
        "Ad area: "
        f"{result['area_pixels']} px "
        f"({result['area_percent']:.3f}% of the image)"
    )
    print(
        "Bounding box: "
        f"x={result['bbox']['x']}, y={result['bbox']['y']}, "
        f"width={result['bbox']['width']}, height={result['bbox']['height']}"
    )
    print(f"Output image: {result['output_image_path']}")

    dimensions = result["dimensions"]
    if result["shape"] == "l":
        print(f"Corner: {dimensions['corner']}")
        for arm_name, arm_size in dimensions.items():
            if arm_name == "corner":
                continue
            pretty_name = arm_name.replace("_", " ")
            print(
                f"{pretty_name.title()}: "
                f"{arm_size['width_px']} x {arm_size['height_px']} px"
            )
    else:
        print(f"Width: {dimensions['width_px']} px")
        print(f"Height: {dimensions['height_px']} px")

    if result.get("warning_reasons"):
        for reason in result["warning_reasons"]:
            print(f"Warning detail: {reason}")

    if "warning" in result:
        print(f"Warning: {result['warning']}")


def process_single_image(image_path, color_hint, shape_hint, tolerance, output_path):
    cleanup_output_variants(
        image_path=image_path,
        output_dir=output_path.parent,
        keep_path=output_path,
    )
    result, render_data = detect_ad(
        image_path=image_path,
        color_hint=color_hint,
        shape_hint=shape_hint,
        tolerance=tolerance,
    )
    save_annotated_image(
        image_rgb=render_data["image_rgb"],
        component_mask=render_data["component_mask"],
        result=result,
        edge_pair=render_data["edge_pair"],
        output_path=output_path,
    )
    result["output_image_path"] = str(output_path)
    return result


def process_image_directory(source_dir, output_dir, color_hint, shape_hint, tolerance):
    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {source_dir}")

    image_paths = find_source_files(source_dir, "images")
    summary = {
        "source": "images",
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "processed_count": len(image_paths),
        "success_count": 0,
        "no_detection_count": 0,
        "error_count": 0,
        "results": [],
        "no_detections": [],
        "errors": [],
    }

    for image_path in image_paths:
        output_path = detection_output_path(image_path, output_dir)
        try:
            result = process_single_image(
                image_path=image_path,
                color_hint=color_hint,
                shape_hint=shape_hint,
                tolerance=tolerance,
                output_path=output_path,
            )
            summary["results"].append(result)
            summary["success_count"] += 1
        except NoAdsDetectedError as exc:
            no_ad_path = no_ad_output_path(image_path, output_dir)
            cleanup_output_variants(
                image_path=image_path,
                output_dir=output_dir,
                keep_path=no_ad_path,
            )
            save_no_ad_image(image_path=image_path, output_path=no_ad_path)
            summary["no_detections"].append(
                {
                    "image_path": str(image_path),
                    "report": "No ads dection",
                    "reason": exc.reason,
                    "output_image_path": str(no_ad_path),
                }
            )
            summary["no_detection_count"] += 1
        except Exception as exc:
            summary["errors"].append(
                {
                    "image_path": str(image_path),
                    "error": str(exc),
                }
            )
            summary["error_count"] += 1
        finally:
            progress_current = (
                summary["success_count"]
                + summary["no_detection_count"]
                + summary["error_count"]
            )
            print_progress("Processing images", progress_current, len(image_paths))

    finish_progress(len(image_paths))

    return summary


def process_video_directory(source_dir, output_dir, color_hint, shape_hint, tolerance, frame_interval_seconds):
    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {source_dir}")

    video_paths = find_source_files(source_dir, "videos")
    summary = {
        "source": "videos",
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "frame_interval_seconds": frame_interval_seconds,
        "processed_count": len(video_paths),
        "success_count": 0,
        "error_count": 0,
        "total_extracted_frames": 0,
        "total_detected_frames": 0,
        "total_no_detection_frames": 0,
        "total_frame_errors": 0,
        "total_warning_frames": 0,
        "videos": [],
        "errors": [],
    }

    for video_path in video_paths:
        video_output_dir = output_dir / video_path.stem
        frames_dir = video_output_dir / "frames"
        detections_dir = video_output_dir / "detections"

        try:
            extracted_frames = extract_frames_from_video(
                video_path=video_path,
                frames_dir=frames_dir,
                frame_interval_seconds=frame_interval_seconds,
            )
            video_summary = {
                "video_path": str(video_path),
                "frames_dir": str(frames_dir),
                "detections_dir": str(detections_dir),
                "frame_interval_seconds": frame_interval_seconds,
                "extracted_frame_count": len(extracted_frames),
                "success_count": 0,
                "no_detection_count": 0,
                "error_count": 0,
                "no_detections": [],
                "results": [],
                "errors": [],
            }

            for frame_info in extracted_frames:
                frame_path = frame_info["frame_path"]
                output_path = detection_output_path(frame_path, detections_dir)
                try:
                    result = process_single_image(
                        image_path=frame_path,
                        color_hint=color_hint,
                        shape_hint=shape_hint,
                        tolerance=tolerance,
                        output_path=output_path,
                    )
                    result["frame_index"] = frame_info["frame_index"]
                    result["timestamp_seconds"] = frame_info["timestamp_seconds"]
                    video_summary["results"].append(result)
                    video_summary["success_count"] += 1
                except NoAdsDetectedError as exc:
                    no_ad_path = no_ad_output_path(frame_path, detections_dir)
                    cleanup_output_variants(
                        image_path=frame_path,
                        output_dir=detections_dir,
                        keep_path=no_ad_path,
                    )
                    save_no_ad_image(image_path=frame_path, output_path=no_ad_path)
                    video_summary["no_detections"].append(
                        {
                            "frame_path": str(frame_path),
                            "frame_index": frame_info["frame_index"],
                            "timestamp_seconds": frame_info["timestamp_seconds"],
                            "report": "No ads dection",
                            "reason": exc.reason,
                            "output_image_path": str(no_ad_path),
                        }
                    )
                    video_summary["no_detection_count"] += 1
                except Exception as exc:
                    video_summary["errors"].append(
                        {
                            "frame_path": str(frame_path),
                            "frame_index": frame_info["frame_index"],
                            "timestamp_seconds": frame_info["timestamp_seconds"],
                            "error": str(exc),
                        }
                    )
                    video_summary["error_count"] += 1
                finally:
                    processed_frames = (
                        video_summary["success_count"]
                        + video_summary["no_detection_count"]
                        + video_summary["error_count"]
                    )
                    print_progress(
                        label=f"Detecting {video_path.stem}",
                        current=processed_frames,
                        total=len(extracted_frames),
                    )

            finish_progress(len(extracted_frames))
            video_summary = apply_video_frame_warnings(video_summary)
            summary["videos"].append(video_summary)
            summary["total_extracted_frames"] += video_summary["extracted_frame_count"]
            summary["total_detected_frames"] += video_summary["success_count"]
            summary["total_no_detection_frames"] += video_summary["no_detection_count"]
            summary["total_frame_errors"] += video_summary["error_count"]
            summary["total_warning_frames"] += video_summary["warning_count"]
            if video_summary["error_count"] == 0:
                summary["success_count"] += 1
            else:
                summary["error_count"] += 1
        except Exception as exc:
            summary["errors"].append(
                {
                    "video_path": str(video_path),
                    "error": str(exc),
                }
            )
            summary["error_count"] += 1
        finally:
            processed_videos = summary["success_count"] + summary["error_count"]
            print_progress("Processing videos", processed_videos, len(video_paths))

    finish_progress(len(video_paths))

    return summary


def print_image_batch_human_readable(summary):
    print(f"Source: {summary['source']}")
    print(f"Input folder: {summary['input_dir']}")
    print(f"Output folder: {summary['output_dir']}")
    print(
        "Processed: "
        f"{summary['processed_count']} | "
        f"Success: {summary['success_count']} | "
        f"No ads: {summary['no_detection_count']} | "
        f"Errors: {summary['error_count']}"
    )

    if not summary["results"] and not summary["errors"] and not summary["no_detections"]:
        print("No supported image files found.")
        return

    for index, result in enumerate(summary["results"], start=1):
        print()
        print(f"--- Result {index} ---")
        print_human_readable(result)

    for index, item in enumerate(summary["no_detections"], start=1):
        print()
        print(f"--- No Detection {index} ---")
        print(f"Image: {item['image_path']}")
        print(f"Report: {item['report']}")
        print(f"Reason: {item['reason']}")
        print(f"Output image: {item['output_image_path']}")

    for index, error_info in enumerate(summary["errors"], start=1):
        print()
        print(f"--- Error {index} ---")
        print(f"Image: {error_info['image_path']}")
        print(f"Error: {error_info['error']}")


def print_video_batch_human_readable(summary):
    print(f"Source: {summary['source']}")
    print(f"Input folder: {summary['input_dir']}")
    print(f"Output folder: {summary['output_dir']}")
    print(f"Frame interval: {summary['frame_interval_seconds']} s")
    print(
        "Processed videos: "
        f"{summary['processed_count']} | "
        f"Success: {summary['success_count']} | "
        f"Errors: {summary['error_count']}"
    )
    print(
        "Frames: "
        f"Extracted {summary['total_extracted_frames']} | "
        f"Detected {summary['total_detected_frames']} | "
        f"No ads {summary['total_no_detection_frames']} | "
        f"Warnings {summary['total_warning_frames']} | "
        f"Frame errors {summary['total_frame_errors']}"
    )

    if not summary["videos"] and not summary["errors"]:
        print("No supported video files found.")
        return

    for index, video_summary in enumerate(summary["videos"], start=1):
        print()
        print(f"--- Video {index} ---")
        print(f"Video: {video_summary['video_path']}")
        print(f"Frames folder: {video_summary['frames_dir']}")
        print(f"Detections folder: {video_summary['detections_dir']}")
        print(
            "Extracted frames: "
            f"{video_summary['extracted_frame_count']} | "
            f"Success: {video_summary['success_count']} | "
            f"No ads: {video_summary['no_detection_count']} | "
            f"Warnings: {video_summary['warning_count']} | "
            f"Errors: {video_summary['error_count']}"
        )

        for frame_index, result in enumerate(video_summary["results"], start=1):
            print()
            if result.get("is_warning"):
                print(
                    f"!!! WARNING FRAME {frame_index} "
                    f"(t={result['timestamp_seconds']:.3f}s, idx={result['frame_index']}) !!!"
                )
            else:
                print(
                    f"--- Frame {frame_index} "
                    f"(t={result['timestamp_seconds']:.3f}s, idx={result['frame_index']}) ---"
                )
            print_human_readable(result)

        for frame_index, error_info in enumerate(video_summary["errors"], start=1):
            print()
            print(
                f"--- Frame Error {frame_index} "
                f"(t={error_info['timestamp_seconds']:.3f}s, idx={error_info['frame_index']}) ---"
            )
            print(f"Frame: {error_info['frame_path']}")
            print(f"Error: {error_info['error']}")

        for frame_index, item in enumerate(video_summary["no_detections"], start=1):
            print()
            print(
                f"--- No Detection Frame {frame_index} "
                f"(t={item['timestamp_seconds']:.3f}s, idx={item['frame_index']}) ---"
            )
            print(f"Frame: {item['frame_path']}")
            print(f"Report: {item['report']}")
            print(f"Reason: {item['reason']}")
            print(f"Output image: {item['output_image_path']}")

    for index, error_info in enumerate(summary["errors"], start=1):
        print()
        print(f"--- Video Error {index} ---")
        print(f"Video: {error_info['video_path']}")
        print(f"Error: {error_info['error']}")


def print_batch_human_readable(summary):
    if summary["source"] == "videos":
        print_video_batch_human_readable(summary)
        return
    print_image_batch_human_readable(summary)


def main():
    args = parse_args()
    source_dir = input_source_dir(args.source)
    output_dir = output_source_dir(args.source)

    if args.source == "videos":
        summary = process_video_directory(
            source_dir=source_dir,
            output_dir=output_dir,
            color_hint=args.color,
            shape_hint=args.shape,
            tolerance=args.tolerance,
            frame_interval_seconds=args.frame_interval,
        )
    else:
        summary = process_image_directory(
            source_dir=source_dir,
            output_dir=output_dir,
            color_hint=args.color,
            shape_hint=args.shape,
            tolerance=args.tolerance,
        )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_batch_human_readable(summary)


if __name__ == "__main__":
    main()
