from __future__ import annotations

"""Export the three Panel E localization comparisons as transparent PNG files.

The source file selection, complete/imputed row inclusion, condition filtering,
ground-truth coordinates, participant pairing, paired permutation test, and Holm
correction are reused from the original Panel E plotting script. This revision only
changes the error unit to percent of the interaction-window diagonal and exports one
minimal axis-labelled plot per facial feature.
"""

import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

import plot_residual_vision_localization_panelE_participant_level as original
from plot_emoji_localization_publication_ready import (
    ERROR_Y_MAX_PERCENT,
    ERROR_Y_TICKS_PERCENT,
    EXPORT_DPI,
    FEATURE_ORDER,
    LOCALIZATION_WINDOW_DIAGONAL_PX,
    LOCALIZATION_WINDOW_HEIGHT_PX,
    LOCALIZATION_WINDOW_WIDTH_PX,
    font_pt,
    mm_to_px,
    pt_to_px,
)


SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SOURCE_DIR.parent
SUMMARY_CSV = SOURCE_DIR / "panelE_localization_error_by_feature_participant_means_percent_complete13.csv"

OUTPUT_FILES = {
    "L": OUTPUT_DIR / "E1_left_eye_interaction_error_percent.png",
    "R": OUTPUT_DIR / "E2_right_eye_interaction_error_percent.png",
    "M": OUTPUT_DIR / "E3_mouth_interaction_error_percent.png",
}

# Layout-only parameters. These can be adjusted without changing the analysis.
# Narrow, portrait-like individual panels for the three-panel Panel E layout.
FIG_WIDTH_MM = 32.0
FIG_HEIGHT_MM = 42.0
PLOT_LEFT_MM = 8.0
PLOT_RIGHT_MM = 30.5
PLOT_TOP_MM = 1.5
PLOT_BOTTOM_MM = 39.0

# Preserve the original Panel E palette and encodings.
VISUAL_ONLY_COLOR = original.VISUAL_ONLY_COLOR
VISUAL_IHF_COLOR = original.VISUAL_IHF_COLOR
VISUAL_ONLY_FILL = original.VISUAL_ONLY_FILL
VISUAL_IHF_FILL = original.VISUAL_IHF_FILL
AXIS_COLOR = original.AXIS_COLOR
TEXT_COLOR = original.TEXT_COLOR
PAIR_COLOR = (210, 210, 210)
TRANSPARENT = (255, 255, 255, 0)

FONT_AXIS = font_pt(7.5, False)
FONT_TICK = font_pt(7.0, False)


def interaction_error_percent(distance_px: float) -> float:
    """Euclidean localization error as a percentage of the window diagonal."""
    return distance_px / LOCALIZATION_WINDOW_DIAGONAL_PX * 100.0


def read_percent_rows() -> list[dict[str, object]]:
    """Reuse the original data pipeline, replacing only the distance unit."""
    source_csv = original.choose_source_csv()
    fieldnames, rows = original.read_rows(source_csv)
    ground_truth = original.read_gt()
    long_rows, _ = original.wide_rows_to_long(rows, fieldnames, ground_truth)

    percent_rows: list[dict[str, object]] = []
    for row in long_rows:
        distance_px = math.hypot(
            float(row["marker_window_x_px"]) - float(row["target_window_x_px"]),
            float(row["marker_window_y_px"]) - float(row["target_window_y_px"]),
        )
        converted = dict(row)
        converted["interaction_error_percent"] = interaction_error_percent(distance_px)
        percent_rows.append(converted)
    return percent_rows


def participant_level_means_percent(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["participant_id"]),
            str(row["condition"]),
            str(row["feature"]),
        )
        grouped.setdefault(key, []).append(float(row["interaction_error_percent"]))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def box_stats(values: list[float]) -> dict[str, float]:
    """Match the original panel: box=IQR, centre=median, whiskers=min/max."""
    return {
        "min": min(values),
        "q1": quantile(values, 0.25),
        "median": statistics.median(values),
        "q3": quantile(values, 0.75),
        "max": max(values),
    }


def draw_boxplot(
    draw: ImageDraw.ImageDraw,
    x: float,
    values: list[float],
    y_position,
    fill: tuple[int, int, int],
) -> None:
    stats = box_stats(values)
    box_width = mm_to_px(9.0)
    cap_width = mm_to_px(6.0)
    outline_width = max(1, round(pt_to_px(0.7)))
    draw.line(
        (x, y_position(stats["min"]), x, y_position(stats["max"])),
        fill=AXIS_COLOR,
        width=outline_width,
    )
    for key in ("min", "max"):
        y = y_position(stats[key])
        draw.line(
            (x - cap_width / 2, y, x + cap_width / 2, y),
            fill=AXIS_COLOR,
            width=outline_width,
        )
    draw.rectangle(
        (
            x - box_width / 2,
            y_position(stats["q3"]),
            x + box_width / 2,
            y_position(stats["q1"]),
        ),
        fill=fill,
        outline=AXIS_COLOR,
        width=outline_width,
    )
    draw.line(
        (
            x - box_width / 2,
            y_position(stats["median"]),
            x + box_width / 2,
            y_position(stats["median"]),
        ),
        fill=(25, 25, 25),
        width=max(1, round(pt_to_px(0.7))),
    )


def draw_rotated_y_label(canvas: Image.Image, x: float, y: float) -> None:
    text = "Interaction error (%)"
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), TRANSPARENT))
    bbox = probe.textbbox((0, 0), text, font=FONT_AXIS)
    layer = Image.new(
        "RGBA",
        (bbox[2] - bbox[0] + round(mm_to_px(3)), bbox[3] - bbox[1] + round(mm_to_px(3))),
        TRANSPARENT,
    )
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (layer.width / 2, layer.height / 2),
        text,
        font=FONT_AXIS,
        fill=TEXT_COLOR,
        anchor="mm",
    )
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(
        layer,
        (round(x - layer.width / 2), round(y - layer.height / 2)),
    )


def draw_feature_plot(
    visual_only: list[float],
    visual_ihf: list[float],
) -> Image.Image:
    """Draw only data plus the essential axes, ticks, and y-axis unit."""
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), TRANSPARENT)
    draw = ImageDraw.Draw(canvas)

    left = mm_to_px(PLOT_LEFT_MM)
    right = mm_to_px(PLOT_RIGHT_MM)
    top = mm_to_px(PLOT_TOP_MM)
    bottom = mm_to_px(PLOT_BOTTOM_MM)
    axis_width = max(1, round(pt_to_px(0.55)))

    def y_position(value: float) -> float:
        return bottom - value / ERROR_Y_MAX_PERCENT * (bottom - top)

    for tick in ERROR_Y_TICKS_PERCENT:
        y = y_position(float(tick))
        draw.line(
            (left - mm_to_px(0.8), y, left, y),
            fill=AXIS_COLOR,
            width=axis_width,
        )
        draw.text(
            (left - mm_to_px(1.25), y),
            str(tick),
            font=FONT_TICK,
            fill=TEXT_COLOR,
            anchor="rm",
        )
    draw.line((left, top, left, bottom), fill=AXIS_COLOR, width=axis_width)
    draw.line((left, bottom, right, bottom), fill=AXIS_COLOR, width=axis_width)

    # Keep both groups clear within the narrow portrait panel.
    x_visual = left + 0.26 * (right - left)
    x_ihf = left + 0.79 * (right - left)
    draw_boxplot(draw, x_visual, visual_only, y_position, VISUAL_ONLY_FILL)
    draw_boxplot(draw, x_ihf, visual_ihf, y_position, VISUAL_IHF_FILL)

    # Same deterministic offset is used for both conditions to preserve pairing.
    jitter_mm = [-2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
    dot_radius = mm_to_px(0.65)
    pair_width = max(1, round(pt_to_px(0.25)))
    outline_width = max(1, round(pt_to_px(0.25)))
    for index, (only_value, ihf_value) in enumerate(zip(visual_only, visual_ihf)):
        offset = mm_to_px(jitter_mm[index % len(jitter_mm)])
        x1 = x_visual + offset
        x2 = x_ihf + offset
        y1 = y_position(only_value)
        y2 = y_position(ihf_value)
        draw.line((x1, y1, x2, y2), fill=PAIR_COLOR, width=pair_width)
        draw.ellipse(
            (x1 - dot_radius, y1 - dot_radius, x1 + dot_radius, y1 + dot_radius),
            fill=VISUAL_ONLY_COLOR,
            outline=(255, 255, 255),
            width=outline_width,
        )
        draw.ellipse(
            (x2 - dot_radius, y2 - dot_radius, x2 + dot_radius, y2 + dot_radius),
            fill=VISUAL_IHF_COLOR,
            outline=(255, 255, 255),
            width=outline_width,
        )

    draw_rotated_y_label(
        canvas,
        mm_to_px(2.0),
        (top + bottom) / 2,
    )
    return canvas


def write_percent_summary(
    participant_means: dict[tuple[str, str, str], float],
    raw_p: dict[str, float],
    adjusted_p: dict[str, float],
) -> None:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_ORDER:
        participants, visual_only, visual_ihf = original.paired_values(
            participant_means,
            feature,
        )
        for participant, only_value, ihf_value in zip(
            participants,
            visual_only,
            visual_ihf,
        ):
            rows.append(
                {
                    "participant_id": participant,
                    "feature": feature,
                    "visual_only_interaction_error_percent": f"{only_value:.6f}",
                    "visual_ihf_interaction_error_percent": f"{ihf_value:.6f}",
                    "visual_only_minus_visual_ihf_percent": f"{only_value - ihf_value:.6f}",
                    "raw_p_for_feature": f"{raw_p[feature]:.8f}",
                    "holm_adjusted_p_for_feature": f"{adjusted_p[feature]:.8f}",
                    "normalization": "Euclidean pixel distance / interaction-window diagonal * 100",
                    "window_width_px": f"{LOCALIZATION_WINDOW_WIDTH_PX:.0f}",
                    "window_height_px": f"{LOCALIZATION_WINDOW_HEIGHT_PX:.0f}",
                    "window_diagonal_px": f"{LOCALIZATION_WINDOW_DIAGONAL_PX:.6f}",
                }
            )

    fieldnames = list(rows[0])
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_FILES.values():
        if path.exists():
            path.unlink()


def main() -> None:
    percent_rows = read_percent_rows()
    participant_means = participant_level_means_percent(percent_rows)

    raw_p: dict[str, float] = {}
    paired: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for feature in FEATURE_ORDER:
        values = original.paired_values(participant_means, feature)
        participants, visual_only, visual_ihf = values
        if len(participants) != 13:
            raise ValueError(f"Expected 13 paired participants for {feature}; found {len(participants)}")
        paired[feature] = values
        raw_p[feature] = original.paired_permutation_pvalue(visual_only, visual_ihf)
    adjusted_p = original.holm_bonferroni(raw_p)

    prepare_output_directory()
    for feature in FEATURE_ORDER:
        _, visual_only, visual_ihf = paired[feature]
        image = draw_feature_plot(visual_only, visual_ihf)
        image.save(OUTPUT_FILES[feature], format="PNG", dpi=(EXPORT_DPI, EXPORT_DPI))

    write_percent_summary(participant_means, raw_p, adjusted_p)

    print(
        "interaction_window="
        f"{LOCALIZATION_WINDOW_WIDTH_PX:.0f}x{LOCALIZATION_WINDOW_HEIGHT_PX:.0f}px; "
        f"diagonal={LOCALIZATION_WINDOW_DIAGONAL_PX:.6f}px"
    )
    for feature in FEATURE_ORDER:
        participants, visual_only, visual_ihf = paired[feature]
        print(
            f"{feature}: n={len(participants)}; "
            f"visual_only_mean={statistics.mean(visual_only):.6f}%; "
            f"visual_ihf_mean={statistics.mean(visual_ihf):.6f}%; "
            f"raw_p={raw_p[feature]:.8f}; "
            f"holm_p={adjusted_p[feature]:.8f}"
        )
    print(f"summary={SUMMARY_CSV}")
    for feature in FEATURE_ORDER:
        print(f"saved={OUTPUT_FILES[feature]}")


if __name__ == "__main__":
    main()
