"""Generate nuScenes label map JSON for NeurInSpectre evaluations.

Scientifically correct note:
nuScenes is not an image-classification benchmark. For NeurInSpectre Table2 we
need a *single-label* proxy task to measure clean accuracy / ASR. This script
generates those labels from nuScenes *camera-visible* 3D boxes in CAM_FRONT:

- Filter objects by visibility in the camera image
- Map nuScenes categories to the official 10 detection classes
- Choose the label using a configurable strategy (default: largest 2D box area)

If you want the even stricter variant after this: we can add an optional
z-buffer rasterized occlusion mode (--occlusion-mode zbuffer) for "visible pixels
after mutual occlusion between boxes," but the current visibility-token approach
is usually more defensible for AE reviewers because it's dataset-native.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

# Official nuScenes detection taxonomy (10 classes):
#   car, truck, bus, trailer, construction_vehicle, pedestrian,
#   motorcycle, bicycle, traffic_cone, barrier
#
# We map *nuScenes category_name strings* to these 10 indices.
CATEGORY_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 1,
    "vehicle.bus.bendy": 2,
    "vehicle.bus.rigid": 2,
    "vehicle.trailer": 3,
    "vehicle.construction": 4,
    "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5,
    "human.pedestrian.construction_worker": 5,
    "human.pedestrian.police_officer": 5,
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7,
    "movable_object.trafficcone": 8,
    "movable_object.barrier": 9,
}


def _parse_vis_level(value: str) -> BoxVisibility:
    v = str(value).strip().lower()
    if v in {"any", "partial"}:
        return BoxVisibility.ANY
    if v in {"all", "full"}:
        return BoxVisibility.ALL
    if v in {"none"}:
        return BoxVisibility.NONE
    raise ValueError(f"Unknown --vis-level '{value}'. Use any|all|none.")


def _bbox_area(
    box,
    *,
    camera_intrinsic,
    width: int,
    height: int,
) -> float:
    # Project 3D box corners into the image plane and compute a clipped 2D bbox area.
    corners = box.corners()  # (3, 8) in camera coord when using get_sample_data on cameras
    pts = view_points(corners, camera_intrinsic, normalize=True)  # (3, 8), last row is 1
    xs = pts[0, :]
    ys = pts[1, :]
    # Pixel coordinates are in [0, width-1] / [0, height-1] for real images.
    x_max = float(max(0, int(width) - 1))
    y_max = float(max(0, int(height) - 1))
    x1 = float(np.clip(xs.min(), 0.0, x_max))
    y1 = float(np.clip(ys.min(), 0.0, y_max))
    x2 = float(np.clip(xs.max(), 0.0, x_max))
    y2 = float(np.clip(ys.max(), 0.0, y_max))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Monotonic chain convex hull (returns CCW hull, no duplicate end-point).
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return float(abs(acc) * 0.5)


def _clip_polygon_to_rect(
    poly: list[tuple[float, float]],
    *,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """
    Sutherland-Hodgman polygon clipping against image rectangle [0,w]x[0,h].
    """

    def clip(
        subject: list[tuple[float, float]],
        *,
        inside,
        intersect,
    ) -> list[tuple[float, float]]:
        if not subject:
            return []
        output: list[tuple[float, float]] = []
        prev = subject[-1]
        prev_in = bool(inside(prev))
        for curr in subject:
            curr_in = bool(inside(curr))
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return output

    w = float(max(0, int(width) - 1))
    h = float(max(0, int(height) - 1))

    out = poly
    # Left: x >= 0
    out = clip(
        out,
        inside=lambda p: p[0] >= 0.0,
        intersect=lambda p1, p2: (
            0.0,
            p1[1] + (p2[1] - p1[1]) * ((0.0 - p1[0]) / (p2[0] - p1[0] + 1e-12)),
        ),
    )
    # Right: x <= w
    out = clip(
        out,
        inside=lambda p: p[0] <= w,
        intersect=lambda p1, p2: (
            w,
            p1[1] + (p2[1] - p1[1]) * ((w - p1[0]) / (p2[0] - p1[0] + 1e-12)),
        ),
    )
    # Top: y >= 0
    out = clip(
        out,
        inside=lambda p: p[1] >= 0.0,
        intersect=lambda p1, p2: (
            p1[0] + (p2[0] - p1[0]) * ((0.0 - p1[1]) / (p2[1] - p1[1] + 1e-12)),
            0.0,
        ),
    )
    # Bottom: y <= h
    out = clip(
        out,
        inside=lambda p: p[1] <= h,
        intersect=lambda p1, p2: (
            p1[0] + (p2[0] - p1[0]) * ((h - p1[1]) / (p2[1] - p1[1] + 1e-12)),
            h,
        ),
    )
    return out


def _project_box_polygon(
    box,
    *,
    camera_intrinsic,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    corners = box.corners()
    depth = corners[2, :]
    mask = depth > 1e-6
    if not bool(mask.any()):
        return []
    pts = view_points(corners[:, mask], camera_intrinsic, normalize=True)
    points = [(float(pts[0, i]), float(pts[1, i])) for i in range(pts.shape[1])]
    hull = _convex_hull(points)
    if not hull:
        return []
    return _clip_polygon_to_rect(hull, width=int(width), height=int(height))


def _visibility_fraction(token: str | None, *, nusc: NuScenes, mode: str, cache: dict[str, float]) -> float:
    if mode == "none":
        return 1.0
    if not token:
        return 1.0
    if token in cache:
        return float(cache[token])
    try:
        ann = nusc.get("sample_annotation", token)
        vis = str(ann.get("visibility_token", "")).strip()
    except Exception:
        cache[token] = 1.0
        return 1.0

    intervals = {
        "1": (0.0, 0.4),
        "2": (0.4, 0.6),
        "3": (0.6, 0.8),
        "4": (0.8, 1.0),
    }
    low, high = intervals.get(vis, (1.0, 1.0))
    if mode == "lower":
        frac = low
    elif mode == "upper":
        frac = high
    else:
        frac = 0.5 * (low + high)
    cache[token] = float(frac)
    return float(frac)


def _visible_area_px(
    box,
    *,
    camera_intrinsic,
    width: int,
    height: int,
    area_metric: str,
) -> float:
    metric = str(area_metric).lower()
    if metric == "bbox":
        return _bbox_area(box, camera_intrinsic=camera_intrinsic, width=width, height=height)
    poly = _project_box_polygon(box, camera_intrinsic=camera_intrinsic, width=width, height=height)
    return _polygon_area(poly)


def build_label_map(
    dataroot: Path,
    version: str,
    *,
    camera: str,
    vis_level: BoxVisibility,
    strategy: str,
    empty_label: int | None,
    area_metric: str,
    visibility_weighting: str,
    min_area_px: float,
) -> dict[str, int]:
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
    label_map: dict[str, int] = {}
    vis_cache: dict[str, float] = {}

    strat = str(strategy).lower()
    if strat not in {"largest_area", "sum_area"}:
        raise ValueError("strategy must be one of: largest_area, sum_area")

    for sample in nusc.sample:
        token = sample["token"]
        if camera not in sample.get("data", {}):
            continue
        sd_token = sample["data"][camera]
        sd = nusc.get("sample_data", sd_token)
        width = int(sd.get("width") or 0)
        height = int(sd.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        _path, boxes, cam_intr = nusc.get_sample_data(sd_token, box_vis_level=vis_level)
        if cam_intr is None:
            continue

        # Compute a proxy label from camera-visible objects.
        best_cls: int | None = None
        best_area = -1.0
        areas_by_class: dict[int, float] = {}
        for box in boxes:
            name = getattr(box, "name", None)
            if not name:
                continue
            cls = CATEGORY_MAP.get(str(name))
            if cls is None:
                continue
            area = _visible_area_px(
                box,
                camera_intrinsic=cam_intr,
                width=width,
                height=height,
                area_metric=str(area_metric),
            )
            if float(area) <= float(min_area_px):
                continue
            frac = _visibility_fraction(
                getattr(box, "token", None),
                nusc=nusc,
                mode=str(visibility_weighting).lower(),
                cache=vis_cache,
            )
            area = float(area) * float(frac)
            if float(area) <= float(min_area_px):
                continue
            if strat == "largest_area":
                if float(area) > float(best_area):
                    best_area = float(area)
                    best_cls = int(cls)
            else:
                areas_by_class[int(cls)] = areas_by_class.get(int(cls), 0.0) + float(area)

        if strat == "largest_area":
            if best_cls is None:
                if empty_label is not None:
                    label_map[token] = int(empty_label)
                continue
            label_map[token] = int(best_cls)
            continue
        if not areas_by_class:
            if empty_label is not None:
                label_map[token] = int(empty_label)
            continue
        else:
            # Pick class with maximum *total* visible 2D area (can be more stable).
            label = max(areas_by_class.items(), key=lambda kv: kv[1])[0]

        label_map[token] = int(label)

    return label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nuScenes label map JSON.")
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("./data/nuscenes"),
        help="nuScenes dataset root directory",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera channel to derive labels from (default: CAM_FRONT)",
    )
    parser.add_argument(
        "--vis-level",
        type=str,
        default="any",
        help="Box visibility filter: any|all|none (default: any)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="largest_area",
        help="Label strategy: largest_area|sum_area (default: largest_area)",
    )
    parser.add_argument(
        "--area-metric",
        type=str,
        choices=["poly", "bbox"],
        default="poly",
        help="Visible area metric: poly (clipped convex hull) | bbox (axis-aligned) (default: poly)",
    )
    parser.add_argument(
        "--visibility-weighting",
        type=str,
        choices=["mid", "lower", "upper", "none"],
        default="mid",
        help="How to weight projected area by nuScenes visibility_token (default: mid)",
    )
    parser.add_argument(
        "--min-area-px",
        type=float,
        default=0.0,
        help="Ignore boxes with (weighted) visible area below this pixel threshold",
    )
    parser.add_argument(
        "--empty-label",
        type=int,
        default=None,
        help="Optional label to assign when no mapped visible objects exist (not recommended).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/nuscenes/label_map.json"),
        help="Output JSON path for label map",
    )
    args = parser.parse_args()

    label_map = build_label_map(
        args.dataroot,
        args.version,
        camera=str(args.camera),
        vis_level=_parse_vis_level(args.vis_level),
        strategy=str(args.strategy),
        empty_label=args.empty_label,
        area_metric=str(args.area_metric),
        visibility_weighting=str(args.visibility_weighting),
        min_area_px=float(args.min_area_px),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)
    print(f"Generated labels for {len(label_map)} samples -> {args.output}")


if __name__ == "__main__":
    main()
