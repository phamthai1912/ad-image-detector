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
WARNING_STABLE_GROUP_MIN_FRAMES = 3
WARNING_AREA_RELATIVE_TOLERANCE = 0.03
WARNING_AREA_ABSOLUTE_TOLERANCE_PIXELS = 500
WARNING_DIMENSION_RELATIVE_TOLERANCE = 0.03
WARNING_DIMENSION_ABSOLUTE_TOLERANCE_PX = 6
MIN_L_PAIR_SCORE = 0.35
MIN_RECTANGULARITY_FOR_RECT = 0.9
FULL_FRAME_AD_THRESHOLD_PERCENT = 90
MIN_AD_AREA_PERCENT = 5.0
MAX_L_AD_AREA_PERCENT = 50.0
MULTICOLOR_BOUNDARY_MARGIN_RATIO = 0.05
MULTICOLOR_BOUNDARY_MERGE_RATIO = 0.02
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class NoAdsDetectedError(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def validate_ad_area_percent(area_percent, shape_name):
    if area_percent < MIN_AD_AREA_PERCENT:
        raise NoAdsDetectedError(
            f"No ads dection: detected ad area is below {MIN_AD_AREA_PERCENT:.0f}% of the frame."
        )
    if shape_name == "l" and area_percent > MAX_L_AD_AREA_PERCENT:
        raise NoAdsDetectedError(
            f"No ads dection: detected L ad area is above {MAX_L_AD_AREA_PERCENT:.0f}% of the frame."
        )
    if area_percent >= FULL_FRAME_AD_THRESHOLD_PERCENT:
        raise NoAdsDetectedError(
            "No ads dection: detected ad covers the whole frame."
        )


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
        "--ad-mode",
        choices=["monochrome", "multicolor"],
        help=(
            "Ad detection mode. If omitted, try monochrome first and "
            "fallback to multicolor when no ad is found."
        ),
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


def smooth_profile(values, kernel_size=9):
    if len(values) == 0:
        return values
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(values, kernel, mode="same")


def boundary_diff_maps(image_rgb):
    image_int = image_rgb.astype(np.int16)
    column_diff = np.mean(
        np.abs(image_int[:, 1:, :] - image_int[:, :-1, :]),
        axis=2,
    )
    row_diff = np.mean(
        np.abs(image_int[1:, :, :] - image_int[:-1, :, :]),
        axis=2,
    )
    return column_diff, row_diff


def boundary_strength_profiles(image_rgb):
    column_diff, row_diff = boundary_diff_maps(image_rgb)
    column_profile = np.percentile(column_diff, 85, axis=0)
    row_profile = np.percentile(row_diff, 85, axis=1)
    return smooth_profile(column_profile), smooth_profile(row_profile)


def boundary_extent_ratio(diff_values, tolerance):
    extent_threshold = max(7.0, float(tolerance) * 0.35)
    return float(np.mean(diff_values >= extent_threshold))


def merge_boundary_positions(candidates, merge_distance):
    if not candidates:
        return []

    groups = []
    current_group = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate[0] - current_group[-1][0] <= merge_distance:
            current_group.append(candidate)
            continue
        groups.append(current_group)
        current_group = [candidate]
    groups.append(current_group)

    merged_positions = []
    for group in groups:
        positions = np.array([item[0] for item in group], dtype=np.float64)
        strengths = np.array([item[1] for item in group], dtype=np.float64)
        merged_positions.append(int(round(np.average(positions, weights=strengths))))
    return merged_positions


def detect_boundary_positions(profile, frame_size, tolerance):
    if len(profile) == 0:
        return []

    margin = max(12, int(round(frame_size * MULTICOLOR_BOUNDARY_MARGIN_RATIO)))
    merge_distance = max(8, int(round(frame_size * MULTICOLOR_BOUNDARY_MERGE_RATIO)))
    threshold = float(np.mean(profile) + np.std(profile) * 0.8 + max(3.0, tolerance * 0.15))

    candidates = [
        (index + 1, float(value))
        for index, value in enumerate(profile)
        if margin < index + 1 < frame_size - margin and value >= threshold
    ]
    if not candidates:
        return []

    return merge_boundary_positions(candidates, merge_distance)


def mask_bbox(component_mask):
    ys, xs = np.where(component_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }


def average_color_hex(image_rgb, component_mask):
    pixels = image_rgb[component_mask.astype(bool)]
    if len(pixels) == 0:
        return "#00FFFF"
    mean_color = tuple(int(round(value)) for value in np.mean(pixels, axis=0).tolist())
    return rgb_to_hex(mean_color)


def make_candidate_from_mask(image_rgb, component_mask, shape_hint, boundary_score=0.0):
    bbox = mask_bbox(component_mask)
    if bbox is None:
        return None

    area_pixels = int(component_mask.sum())
    component_crop = component_mask[
        bbox["y"] : bbox["y"] + bbox["height"],
        bbox["x"] : bbox["x"] + bbox["width"],
    ]
    inferred_shape, edges, edge_pair, rectangularity, best_pair_score = infer_shape_from_component(
        component_crop=component_crop,
        bbox_width=bbox["width"],
        bbox_height=bbox["height"],
        area_pixels=area_pixels,
    )
    if inferred_shape == "unknown":
        return None

    l_balance_score = 0.0
    if inferred_shape == "l" and edge_pair is not None:
        edge_a, edge_b = normalize_l_pair(edge_pair)
        vertical_thickness = edges[edge_a] if edge_a in ("left", "right") else edges[edge_b]
        horizontal_thickness = edges[edge_a] if edge_a in ("top", "bottom") else edges[edge_b]
        if max(vertical_thickness, horizontal_thickness) > 0:
            l_balance_score = min(vertical_thickness, horizontal_thickness) / float(
                max(vertical_thickness, horizontal_thickness)
            )

    return {
        "component_mask": component_mask,
        "bbox": bbox,
        "area_pixels": area_pixels,
        "inferred_shape": inferred_shape,
        "edges": edges,
        "edge_pair": edge_pair,
        "rectangularity": rectangularity,
        "best_pair_score": best_pair_score,
        "selection_score": compute_component_score(
            area_pixels=area_pixels,
            inferred_shape=inferred_shape,
            rectangularity=rectangularity,
            best_pair_score=best_pair_score,
            bbox_width=bbox["width"],
            bbox_height=bbox["height"],
            shape_hint=shape_hint,
        ),
        "detected_color_hex": average_color_hex(image_rgb, component_mask),
        "boundary_score": float(boundary_score),
        "l_balance_score": float(l_balance_score),
    }


def rectangle_mask(image_shape, x0, y0, x1, y1):
    component_mask = np.zeros(image_shape[:2], dtype=bool)
    component_mask[y0:y1, x0:x1] = True
    return component_mask


def l_shape_mask(image_shape, boundary_x, boundary_y, content_corner):
    image_height, image_width = image_shape[:2]
    component_mask = np.ones((image_height, image_width), dtype=bool)
    content_boxes = {
        "top-left": (0, 0, boundary_x, boundary_y),
        "top-right": (boundary_x, 0, image_width, boundary_y),
        "bottom-left": (0, boundary_y, boundary_x, image_height),
        "bottom-right": (boundary_x, boundary_y, image_width, image_height),
    }
    x0, y0, x1, y1 = content_boxes[content_corner]
    component_mask[y0:y1, x0:x1] = False
    return component_mask


def generate_multicolor_candidates(
    image_rgb,
    vertical_boundaries,
    horizontal_boundaries,
    vertical_strengths,
    horizontal_strengths,
    shape_hint,
):
    image_height, image_width = image_rgb.shape[:2]
    candidates = []

    for boundary_x in vertical_boundaries:
        for component_mask in (
            rectangle_mask(image_rgb.shape, 0, 0, boundary_x, image_height),
            rectangle_mask(image_rgb.shape, boundary_x, 0, image_width, image_height),
        ):
            candidate = make_candidate_from_mask(
                image_rgb,
                component_mask,
                shape_hint,
                boundary_score=vertical_strengths.get(boundary_x, 0.0),
            )
            if candidate is not None:
                candidates.append(candidate)

    for boundary_y in horizontal_boundaries:
        for component_mask in (
            rectangle_mask(image_rgb.shape, 0, 0, image_width, boundary_y),
            rectangle_mask(image_rgb.shape, 0, boundary_y, image_width, image_height),
        ):
            candidate = make_candidate_from_mask(
                image_rgb,
                component_mask,
                shape_hint,
                boundary_score=horizontal_strengths.get(boundary_y, 0.0),
            )
            if candidate is not None:
                candidates.append(candidate)

    for boundary_x in vertical_boundaries:
        for boundary_y in horizontal_boundaries:
            for content_corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
                component_mask = l_shape_mask(
                    image_shape=image_rgb.shape,
                    boundary_x=boundary_x,
                    boundary_y=boundary_y,
                    content_corner=content_corner,
                )
                candidate = make_candidate_from_mask(
                    image_rgb,
                    component_mask,
                    shape_hint,
                    boundary_score=(
                        vertical_strengths.get(boundary_x, 0.0)
                        + horizontal_strengths.get(boundary_y, 0.0)
                    ),
                )
                if candidate is not None:
                    candidates.append(candidate)

    return candidates


def build_multicolor_candidates(image_rgb, shape_hint, tolerance):
    image_width = image_rgb.shape[1]
    image_height = image_rgb.shape[0]
    column_diff, row_diff = boundary_diff_maps(image_rgb)
    vertical_profile = smooth_profile(np.percentile(column_diff, 85, axis=0))
    horizontal_profile = smooth_profile(np.percentile(row_diff, 85, axis=1))
    vertical_boundaries = detect_boundary_positions(vertical_profile, image_width, tolerance)
    horizontal_boundaries = detect_boundary_positions(horizontal_profile, image_height, tolerance)
    vertical_strengths = {
        position: float(vertical_profile[position - 1])
        * boundary_extent_ratio(column_diff[:, position - 1], tolerance)
        for position in vertical_boundaries
    }
    horizontal_strengths = {
        position: float(horizontal_profile[position - 1])
        * boundary_extent_ratio(row_diff[position - 1, :], tolerance)
        for position in horizontal_boundaries
    }
    candidates = generate_multicolor_candidates(
        image_rgb=image_rgb,
        vertical_boundaries=vertical_boundaries,
        horizontal_boundaries=horizontal_boundaries,
        vertical_strengths=vertical_strengths,
        horizontal_strengths=horizontal_strengths,
        shape_hint=shape_hint,
    )
    return {
        "candidates": candidates,
        "vertical_boundaries": vertical_boundaries,
        "horizontal_boundaries": horizontal_boundaries,
        "boundary_count": len(vertical_boundaries) + len(horizontal_boundaries),
    }


def is_candidate_area_valid(area_pixels, image_area, shape_name=None):
    if image_area <= 0:
        return True
    area_percent = float(area_pixels) / float(image_area) * 100.0
    if area_percent < MIN_AD_AREA_PERCENT or area_percent >= FULL_FRAME_AD_THRESHOLD_PERCENT:
        return False
    if shape_name == "l" and area_percent > MAX_L_AD_AREA_PERCENT:
        return False
    return True


def choose_multicolor_candidate(candidates, shape_hint, image_area=None):
    if not candidates:
        return None

    if image_area is not None:
        valid_candidates = [
            candidate
            for candidate in candidates
            if is_candidate_area_valid(
                candidate["area_pixels"],
                image_area,
                shape_name=candidate["inferred_shape"],
            )
        ]
        if valid_candidates:
            candidates = valid_candidates

    rect_candidates = [
        candidate
        for candidate in candidates
        if candidate["inferred_shape"] in {"rectangle", "square"}
    ]

    if shape_hint == "l":
        l_candidates = [
            candidate for candidate in candidates if candidate["inferred_shape"] == "l"
        ]
        if not l_candidates:
            return None

        def l_horizontal_arm_penalty(candidate):
            edge_a, edge_b = normalize_l_pair(candidate["edge_pair"])
            horizontal_edge = edge_a if edge_a in ("top", "bottom") else edge_b
            horizontal_ratio = candidate["edges"][horizontal_edge] / float(candidate["bbox"]["height"])
            if 0.18 <= horizontal_ratio <= 0.38:
                return 0.0
            if horizontal_ratio < 0.18:
                return 0.18 - horizontal_ratio
            return horizontal_ratio - 0.38

        best_l = min(
            l_candidates,
            key=lambda candidate: (
                l_horizontal_arm_penalty(candidate),
                -candidate["boundary_score"],
                candidate["area_pixels"],
            ),
        )
        return best_l

    if shape_hint == "auto":
        l_candidates = [
            candidate for candidate in candidates if candidate["inferred_shape"] == "l"
        ]
        if l_candidates and rect_candidates:
            best_l = max(
                l_candidates,
                key=lambda candidate: (
                    candidate["boundary_score"] + 0.1 + (0.15 * candidate.get("l_balance_score", 0.0))
                ),
            )
            slender_rect_candidates = []
            for candidate in rect_candidates:
                bbox_width = candidate["bbox"]["width"]
                bbox_height = candidate["bbox"]["height"]
                aspect_ratio = max(bbox_width, bbox_height) / float(min(bbox_width, bbox_height))
                if aspect_ratio > 2.2:
                    slender_rect_candidates.append((aspect_ratio, candidate))

            if slender_rect_candidates:
                _, best_rect = min(
                    slender_rect_candidates,
                    key=lambda item: (
                        -item[1]["boundary_score"],
                        item[1]["area_pixels"],
                    ),
                )
                best_l_area = max(float(best_l["area_pixels"]), 1.0)
                rect_area_ratio = best_rect["area_pixels"] / best_l_area
                if (
                    rect_area_ratio <= 0.7
                    and best_rect["boundary_score"] >= best_l["boundary_score"] * 0.82
                ):
                    return best_rect

    def priority(candidate):
        inferred_shape = candidate["inferred_shape"]
        area_pixels = candidate["area_pixels"]
        boundary_score = candidate["boundary_score"]
        l_balance_score = candidate.get("l_balance_score", 0.0)

        if shape_hint == "l":
            return (
                0 if inferred_shape == "l" else 1,
                -(boundary_score * (0.5 + l_balance_score)),
                area_pixels,
            )
        if shape_hint == "square":
            return (0 if inferred_shape == "square" else 1, -boundary_score, area_pixels)
        if shape_hint == "rectangle":
            is_match = inferred_shape in {"rectangle", "square"}
            return (0 if is_match else 1, -boundary_score, area_pixels)

        effective_score = boundary_score
        if inferred_shape == "l":
            effective_score += 0.1 + (0.15 * l_balance_score)
        return (-effective_score, area_pixels)

    return min(candidates, key=priority)


def build_l_dimensions_from_candidate(candidate):
    return build_dimensions("l", candidate["bbox"], candidate["edges"], candidate["edge_pair"])


def build_multicolor_l_reference(results):
    l_results = [result for result in results if result["shape"] == "l"]
    if len(l_results) < 2:
        return None

    corners = [result["dimensions"]["corner"] for result in l_results]
    dominant_corner = dominant_value(corners)
    dominant_results = [
        result
        for result in l_results
        if result["dimensions"]["corner"] == dominant_corner
    ]
    if len(dominant_results) < 2:
        return None

    reference = {
        "corner": dominant_corner,
        "area_pixels": numeric_median([result["area_pixels"] for result in dominant_results]),
    }
    metric_values = []
    for result in dominant_results:
        dims = result["dimensions"]
        values = {
            "bottom_arm_height_px": dims["bottom_arm"]["height_px"] if "bottom_arm" in dims else None,
            "top_arm_height_px": dims["top_arm"]["height_px"] if "top_arm" in dims else None,
            "left_arm_width_px": dims["left_arm"]["width_px"] if "left_arm" in dims else None,
            "right_arm_width_px": dims["right_arm"]["width_px"] if "right_arm" in dims else None,
        }
        metric_values.append(values)

    for metric_name in list(metric_values[0].keys()):
        present_values = [item[metric_name] for item in metric_values if item[metric_name] is not None]
        if present_values:
            reference[metric_name] = numeric_median(present_values)

    return reference


def build_stable_multicolor_l_reference(results):
    stable_references = build_stable_warning_references(results)
    l_references = []
    for reference in stable_references:
        metrics = reference.get("metrics", {})
        if reference["shape"] != "l" or "corner" not in metrics:
            continue
        l_references.append(
            {
                "shape": reference["shape"],
                "corner": metrics["corner"],
                "area_pixels": reference["area_pixels"],
                "frame_count": reference.get("frame_count", 0),
                "bottom_arm_height_px": metrics.get("bottom_arm_height_px"),
                "top_arm_height_px": metrics.get("top_arm_height_px"),
                "left_arm_width_px": metrics.get("left_arm_width_px"),
                "right_arm_width_px": metrics.get("right_arm_width_px"),
            }
        )

    if not l_references:
        return None

    return max(
        l_references,
        key=lambda reference: (
            reference["frame_count"],
            -reference["area_pixels"],
        ),
    )


def choose_multicolor_candidate_with_reference(candidates, shape_hint, reference, image_area=None):
    if reference is None:
        return choose_multicolor_candidate(candidates, shape_hint, image_area=image_area)

    base_choice = choose_multicolor_candidate(candidates, shape_hint, image_area=image_area)
    if shape_hint == "auto" and (
        base_choice is None or base_choice["inferred_shape"] != "l"
    ):
        return base_choice

    if image_area is not None:
        valid_candidates = [
            candidate
            for candidate in candidates
            if is_candidate_area_valid(
                candidate["area_pixels"],
                image_area,
                shape_name=candidate["inferred_shape"],
            )
        ]
        if valid_candidates:
            candidates = valid_candidates

    l_candidates = [candidate for candidate in candidates if candidate["inferred_shape"] == "l"]
    if shape_hint not in {"auto", "l"} or not l_candidates:
        return choose_multicolor_candidate(candidates, shape_hint, image_area=image_area)

    def reference_distance(candidate):
        dims = build_l_dimensions_from_candidate(candidate)
        corner = dims["corner"]
        corner_penalty = 0 if corner == reference["corner"] else 1

        horizontal_penalty = 0.0
        if "bottom_arm" in dims and "bottom_arm_height_px" in reference:
            horizontal_penalty = abs(
                dims["bottom_arm"]["height_px"] - reference["bottom_arm_height_px"]
            )
        elif "top_arm" in dims and "top_arm_height_px" in reference:
            horizontal_penalty = abs(
                dims["top_arm"]["height_px"] - reference["top_arm_height_px"]
            )
        else:
            horizontal_penalty = candidate["bbox"]["height"]

        vertical_penalty = 0.0
        if "left_arm" in dims and "left_arm_width_px" in reference:
            vertical_penalty = abs(
                dims["left_arm"]["width_px"] - reference["left_arm_width_px"]
            )
        elif "right_arm" in dims and "right_arm_width_px" in reference:
            vertical_penalty = abs(
                dims["right_arm"]["width_px"] - reference["right_arm_width_px"]
            )
        else:
            vertical_penalty = candidate["bbox"]["width"]

        area_penalty = abs(candidate["area_pixels"] - reference["area_pixels"])
        return (
            corner_penalty,
            horizontal_penalty,
            vertical_penalty,
            area_penalty,
            -candidate["boundary_score"],
            candidate["area_pixels"],
        )

    return min(l_candidates, key=reference_distance)


def l_mask_from_reference(image_shape, corner, vertical_width, horizontal_height):
    image_height, image_width = image_shape[:2]
    mask = np.zeros((image_height, image_width), dtype=bool)

    if corner == "bottom-left":
        mask[:, :vertical_width] = True
        mask[image_height - horizontal_height :, :] = True
        return mask
    if corner == "bottom-right":
        mask[:, image_width - vertical_width :] = True
        mask[image_height - horizontal_height :, :] = True
        return mask
    if corner == "top-left":
        mask[:, :vertical_width] = True
        mask[:horizontal_height, :] = True
        return mask

    mask[:, image_width - vertical_width :] = True
    mask[:horizontal_height, :] = True
    return mask


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
            image_rgb = load_image_rgb(result["image_path"])
            component_mask = build_component_mask_from_result(result, image_rgb.shape)
            edge_pair = None
            if result["shape"] == "l":
                corner_to_edge_pair = {
                    "bottom-left": ("left", "bottom"),
                    "bottom-right": ("right", "bottom"),
                    "top-left": ("left", "top"),
                    "top-right": ("right", "top"),
                }
                edge_pair = corner_to_edge_pair[result["dimensions"]["corner"]]
            save_annotated_image(
                image_rgb=image_rgb,
                component_mask=component_mask,
                result=result,
                edge_pair=edge_pair,
                output_path=flagged_output_path,
            )
            result["output_image_path"] = str(flagged_output_path)
            video_summary["warning_count"] += 1
        else:
            if flagged_output_path.exists():
                flagged_output_path.unlink()

    return video_summary


def rebuild_multicolor_result(image_path, shape_hint, tolerance, reference):
    image_rgb = load_image_rgb(image_path)
    image_height, image_width = image_rgb.shape[:2]
    image_area = image_width * image_height
    multicolor_data = build_multicolor_candidates(
        image_rgb=image_rgb,
        shape_hint=shape_hint,
        tolerance=tolerance,
    )
    candidates = multicolor_data["candidates"]
    best_component = choose_multicolor_candidate_with_reference(
        candidates=candidates,
        shape_hint=shape_hint,
        reference=reference,
        image_area=image_area,
    )
    if best_component is None:
        return None

    component_mask = best_component["component_mask"]
    inferred_shape = best_component["inferred_shape"]
    edge_pair = best_component["edge_pair"]
    bbox = best_component["bbox"]
    area_pixels = best_component["area_pixels"]
    area_percent = float(area_pixels) / float(image_width * image_height) * 100.0
    resolved_shape = resolve_shape(shape_hint, inferred_shape)
    try:
        validate_ad_area_percent(area_percent, resolved_shape)
    except NoAdsDetectedError:
        return None

    result = {
        "image_path": str(image_path),
        "detected_color_hex": best_component["detected_color_hex"],
        "shape": resolved_shape,
        "inferred_shape": inferred_shape,
        "bbox": bbox,
        "area_pixels": int(area_pixels),
        "area_percent": round(area_percent, 3),
        "dimensions": build_dimensions(resolved_shape, bbox, best_component["edges"], edge_pair),
        "tolerance": tolerance,
        "rectangularity": round(best_component["rectangularity"], 4),
        "selection_score": round(best_component["selection_score"], 3),
        "ad_mode": "multicolor",
        "boundary_count": multicolor_data["boundary_count"],
    }
    render_data = {
        "image_rgb": image_rgb,
        "component_mask": component_mask,
        "edge_pair": edge_pair,
    }
    return result, render_data


def snap_multicolor_result_to_l_reference(result, reference):
    if result["shape"] != "l":
        return None

    dimensions = result["dimensions"]
    corner = reference["corner"]
    current_horizontal_height = (
        dimensions["bottom_arm"]["height_px"]
        if "bottom_arm" in dimensions
        else dimensions["top_arm"]["height_px"]
    )
    current_vertical_width = (
        dimensions["left_arm"]["width_px"]
        if "left_arm" in dimensions
        else dimensions["right_arm"]["width_px"]
    )
    target_horizontal_height = reference.get("bottom_arm_height_px")
    if corner.startswith("top"):
        target_horizontal_height = reference.get("top_arm_height_px")
    target_vertical_width = reference.get("left_arm_width_px")
    if corner.endswith("right"):
        target_vertical_width = reference.get("right_arm_width_px")

    if target_horizontal_height is None or target_vertical_width is None:
        return None

    horizontal_delta = abs(current_horizontal_height - target_horizontal_height)
    vertical_delta = abs(current_vertical_width - target_vertical_width)
    if horizontal_delta > 60 or vertical_delta > 40:
        return None

    image_rgb = load_image_rgb(result["image_path"])
    component_mask = l_mask_from_reference(
        image_shape=image_rgb.shape,
        corner=corner,
        vertical_width=int(round(target_vertical_width)),
        horizontal_height=int(round(target_horizontal_height)),
    )
    bbox = mask_bbox(component_mask)
    if bbox is None:
        return None

    area_pixels = int(component_mask.sum())
    image_height, image_width = image_rgb.shape[:2]
    area_percent = float(area_pixels) / float(image_width * image_height) * 100.0
    snapped_result = dict(result)
    snapped_result["detected_color_hex"] = average_color_hex(image_rgb, component_mask)
    snapped_result["bbox"] = bbox
    snapped_result["area_pixels"] = area_pixels
    snapped_result["area_percent"] = round(area_percent, 3)
    snapped_result["dimensions"] = {
        "corner": corner,
    }
    if corner.endswith("left"):
        snapped_result["dimensions"]["left_arm"] = {
            "width_px": int(round(target_vertical_width)),
            "height_px": image_height,
        }
    else:
        snapped_result["dimensions"]["right_arm"] = {
            "width_px": int(round(target_vertical_width)),
            "height_px": image_height,
        }

    if corner.startswith("bottom"):
        snapped_result["dimensions"]["bottom_arm"] = {
            "width_px": image_width,
            "height_px": int(round(target_horizontal_height)),
        }
        edge_pair = ("left", "bottom") if corner == "bottom-left" else ("right", "bottom")
    else:
        snapped_result["dimensions"]["top_arm"] = {
            "width_px": image_width,
            "height_px": int(round(target_horizontal_height)),
        }
        edge_pair = ("left", "top") if corner == "top-left" else ("right", "top")

    render_data = {
        "image_rgb": image_rgb,
        "component_mask": component_mask,
        "edge_pair": edge_pair,
    }
    return snapped_result, render_data


def refine_multicolor_video_results(video_summary, shape_hint, tolerance):
    has_multicolor_results = any(
        result.get("ad_mode") == "multicolor" for result in video_summary["results"]
    )
    if not has_multicolor_results:
        return video_summary

    corner_reference = build_multicolor_l_reference(video_summary["results"])
    if corner_reference is None:
        return video_summary

    for result in video_summary["results"]:
        if result["shape"] != "l":
            continue
        if result["dimensions"]["corner"] == corner_reference["corner"]:
            continue

        rebuilt = rebuild_multicolor_result(
            image_path=Path(result["image_path"]),
            shape_hint=shape_hint,
            tolerance=tolerance,
            reference=corner_reference,
        )
        if rebuilt is None:
            continue

        rebuilt_result, render_data = rebuilt
        output_path = Path(result["output_image_path"])
        rebuilt_result["frame_index"] = result["frame_index"]
        rebuilt_result["timestamp_seconds"] = result["timestamp_seconds"]
        rebuilt_result["output_image_path"] = str(output_path)
        save_annotated_image(
            image_rgb=render_data["image_rgb"],
            component_mask=render_data["component_mask"],
            result=rebuilt_result,
            edge_pair=render_data["edge_pair"],
            output_path=output_path,
        )
        result.clear()
        result.update(rebuilt_result)

    stable_reference = build_stable_multicolor_l_reference(video_summary["results"])
    if stable_reference is None:
        return video_summary

    for result in video_summary["results"]:
        if result["shape"] != "l":
            continue
        rebuilt = snap_multicolor_result_to_l_reference(result, stable_reference)
        if rebuilt is None:
            continue

        rebuilt_result, render_data = rebuilt
        output_path = Path(result["output_image_path"])
        rebuilt_result["frame_index"] = result["frame_index"]
        rebuilt_result["timestamp_seconds"] = result["timestamp_seconds"]
        rebuilt_result["output_image_path"] = str(output_path)
        save_annotated_image(
            image_rgb=render_data["image_rgb"],
            component_mask=render_data["component_mask"],
            result=rebuilt_result,
            edge_pair=render_data["edge_pair"],
            output_path=output_path,
        )
        result.clear()
        result.update(rebuilt_result)

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


def build_component_mask_from_result(result, image_shape):
    image_height, image_width = image_shape[:2]
    mask = np.zeros((image_height, image_width), dtype=bool)

    if result["shape"] == "l":
        dimensions = result["dimensions"]
        corner = dimensions["corner"]
        horizontal_height = (
            dimensions["bottom_arm"]["height_px"]
            if "bottom_arm" in dimensions
            else dimensions["top_arm"]["height_px"]
        )
        vertical_width = (
            dimensions["left_arm"]["width_px"]
            if "left_arm" in dimensions
            else dimensions["right_arm"]["width_px"]
        )
        return l_mask_from_reference(
            image_shape=image_shape,
            corner=corner,
            vertical_width=int(round(vertical_width)),
            horizontal_height=int(round(horizontal_height)),
        )

    bbox = result["bbox"]
    x = bbox["x"]
    y = bbox["y"]
    width = bbox["width"]
    height = bbox["height"]
    mask[y : y + height, x : x + width] = True
    return mask


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
    overlay_alpha = 0.28
    if result.get("is_warning"):
        border_color_bgr = (0, 0, 255)
        overlay_alpha = 0.45

    apply_mask_overlay(
        image_bgr,
        component_mask,
        color_bgr=border_color_bgr,
        alpha=overlay_alpha,
    )
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


def detect_ad_monochrome(image_path, color_hint, shape_hint, tolerance):
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
    validate_ad_area_percent(area_percent, resolved_shape)
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


def detect_ad_multicolor(image_path, shape_hint, tolerance):
    image_rgb = load_image_rgb(image_path)
    image_width = image_rgb.shape[1]
    image_height = image_rgb.shape[0]
    image_area = image_width * image_height

    multicolor_data = build_multicolor_candidates(
        image_rgb=image_rgb,
        shape_hint=shape_hint,
        tolerance=tolerance,
    )
    vertical_boundaries = multicolor_data["vertical_boundaries"]
    horizontal_boundaries = multicolor_data["horizontal_boundaries"]
    if not vertical_boundaries and not horizontal_boundaries:
        raise NoAdsDetectedError(
            "No ads dection: cannot find strong straight boundaries between ad and content."
        )

    candidates = multicolor_data["candidates"]
    best_component = choose_multicolor_candidate(
        candidates,
        shape_hint,
        image_area=image_area,
    )
    if best_component is None:
        raise NoAdsDetectedError(
            "No ads dection: boundaries do not form a valid L, square, or rectangle ad."
        )

    component_mask = best_component["component_mask"]
    inferred_shape = best_component["inferred_shape"]
    edges = best_component["edges"]
    edge_pair = best_component["edge_pair"]
    rectangularity = best_component["rectangularity"]
    resolved_shape = resolve_shape(shape_hint, inferred_shape)
    bbox = best_component["bbox"]
    area_pixels = best_component["area_pixels"]

    area_percent = float(area_pixels) / float(image_width * image_height) * 100.0
    validate_ad_area_percent(area_percent, resolved_shape)

    result = {
        "image_path": str(image_path),
        "detected_color_hex": best_component["detected_color_hex"],
        "shape": resolved_shape,
        "inferred_shape": inferred_shape,
        "bbox": bbox,
        "area_pixels": int(area_pixels),
        "area_percent": round(area_percent, 3),
        "dimensions": build_dimensions(resolved_shape, bbox, edges, edge_pair),
        "tolerance": tolerance,
        "rectangularity": round(rectangularity, 4),
        "selection_score": round(best_component["selection_score"], 3),
        "ad_mode": "multicolor",
        "boundary_count": multicolor_data["boundary_count"],
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


def detect_ad(image_path, color_hint, shape_hint, tolerance, ad_mode):
    if ad_mode == "multicolor":
        return detect_ad_multicolor(
            image_path=image_path,
            shape_hint=shape_hint,
            tolerance=tolerance,
        )
    if ad_mode == "monochrome":
        return detect_ad_monochrome(
            image_path=image_path,
            color_hint=color_hint,
            shape_hint=shape_hint,
            tolerance=tolerance,
        )

    try:
        return detect_ad_monochrome(
            image_path=image_path,
            color_hint=color_hint,
            shape_hint=shape_hint,
            tolerance=tolerance,
        )
    except NoAdsDetectedError:
        return detect_ad_multicolor(
            image_path=image_path,
            shape_hint=shape_hint,
            tolerance=tolerance,
        )


def format_ad_mode_label(ad_mode):
    if ad_mode is None:
        return "auto (monochrome -> multicolor fallback)"
    return ad_mode


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


def process_single_image(image_path, color_hint, shape_hint, tolerance, output_path, ad_mode):
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
        ad_mode=ad_mode,
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


def process_image_directory(source_dir, output_dir, color_hint, shape_hint, tolerance, ad_mode):
    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {source_dir}")

    image_paths = find_source_files(source_dir, "images")
    summary = {
        "source": "images",
        "ad_mode": ad_mode,
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
                ad_mode=ad_mode,
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


def process_video_directory(source_dir, output_dir, color_hint, shape_hint, tolerance, frame_interval_seconds, ad_mode):
    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {source_dir}")

    video_paths = find_source_files(source_dir, "videos")
    summary = {
        "source": "videos",
        "ad_mode": ad_mode,
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
                "ad_mode": ad_mode,
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
                        ad_mode=ad_mode,
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
            video_summary = refine_multicolor_video_results(
                video_summary=video_summary,
                shape_hint=shape_hint,
                tolerance=tolerance,
            )
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
    print(f"Ad mode: {format_ad_mode_label(summary['ad_mode'])}")
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
    print(f"Ad mode: {format_ad_mode_label(summary['ad_mode'])}")
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
            ad_mode=args.ad_mode,
        )
    else:
        summary = process_image_directory(
            source_dir=source_dir,
            output_dir=output_dir,
            color_hint=args.color,
            shape_hint=args.shape,
            tolerance=args.tolerance,
            ad_mode=args.ad_mode,
        )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_batch_human_readable(summary)


if __name__ == "__main__":
    main()
