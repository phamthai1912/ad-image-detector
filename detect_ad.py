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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a single-color ad region and report its size and area percentage."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
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

    if rectangularity < 0.9 and best_pair:
        return "l", edges, best_pair, rectangularity, best_pair_score

    if aspect_gap <= 0.05:
        return "square", edges, best_pair, rectangularity, best_pair_score

    return "rectangle", edges, best_pair, rectangularity, best_pair_score


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
        raise RuntimeError(
            "Cannot find a single-color ad region. "
            "Try passing --color to help the detector."
        )

    label = best_component["label"]
    stats = best_component["stats"]
    x, y, width, height, area_pixels = stats[label]
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

    return result


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

    if "warning" in result:
        print(f"Warning: {result['warning']}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    result = detect_ad(
        image_path=image_path,
        color_hint=args.color,
        shape_hint=args.shape,
        tolerance=args.tolerance,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human_readable(result)


if __name__ == "__main__":
    main()
